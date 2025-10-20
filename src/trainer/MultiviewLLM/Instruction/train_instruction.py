from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from src.utils.MultiviewLLM.Instruction.utils import *
from src.utils.seed_everything import seed_everything


def save_projector(accelerator: Accelerator, projector, save_dir: Path, name: str):
    """
    仅保存 projector（不保存 LLM）。
    会自动在 rank0 进程落盘；其他进程跳过。
    """
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(projector)
        ckpt_path = save_dir / f"{name}.pt"
        torch.save(unwrapped.projector.state_dict(), ckpt_path)
        accelerator.print(f"[Save] Projector saved to: {ckpt_path}")
    accelerator.wait_for_everyone()


def main(config):
    # Set seed for reproducibility
    seed_everything(seed=config['seed'])

    # Set up accelerator and wandb
    project_config = ProjectConfiguration(
        project_dir=config['save_dir'],
        total_limit=5,
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision=config['mixed_precision'],
        log_with="wandb",
        project_config=project_config,
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=config['grad_accumulation_steps'],
    )
    accelerator.init_trackers(
        project_name=config['project'],
        config=config,
        init_kwargs={"wandb":
                         {"name": config['run_name'],
                          "config": config,
                          }
                     }
    )

    # Create tokenize, dataloader, model, optimizer, scheduler
    tokenizer = create_tokenizer(config)
    train_loader, test_loader = create_dataloader(config, tokenizer)
    model, optimizer, warmup_scheduler = create_model_and_optimizer(config, tokenizer, train_loader)

    # ---------- 加载 projector checkpoint ----------
    if config.get('load_checkpoint_path') is not None:
        ckpt_path = Path(config['load_checkpoint_path'])
        if ckpt_path.exists():
            accelerator.print(f"[Load] Loading projector checkpoint from {ckpt_path}")
            # 加载模型权重（兼容 GPU / CPU）
            state_dict = torch.load(ckpt_path, map_location='cpu')
            missing, unexpected = model.projector.load_state_dict(state_dict, strict=False)
            accelerator.print(f"[Load] Missing keys: {missing}, Unexpected keys: {unexpected}")
        else:
            accelerator.print(f"[Warning] Checkpoint not found: {ckpt_path}")

    # Prepare everything with accelerator
    model, optimizer, train_loader, warmup_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        warmup_scheduler,
    )

    # save_projector(accelerator, model, config['save_dir'], f"{config['model_save_name']}_initial")
    # accelerator.print("[Start Training]")

    # Prepare paths for saving models
    global_step = 0
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0.0

        if accelerator.is_main_process:
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        else:
            pbar = None

        for step, batch in enumerate(train_loader):
            # batch = {k: v.to(config['device']) for k, v in batch.items() if k!='original_tags'}
            with accelerator.accumulate(model):
                input_ids = batch['input_ids']
                labels = batch["labels"]
                is_graph = batch['is_graph']
                is_ts = batch['is_ts']
                attn_mask = batch['attn_mask']
                graph_x = batch['graph_x']
                graph_x_pad = batch['graph_x_pad']
                ts_x = batch['ts_x']
                ts_x_pad = batch['ts_x_pad']

                # accelerator.print(f"input_ids shape: {input_ids.shape}")

                output = model(
                    input_ids=input_ids,
                    is_graph=is_graph,
                    is_ts=is_ts,
                    graph_x=graph_x,
                    graph_x_pad=graph_x_pad,
                    ts_x=ts_x,
                    ts_x_pad=ts_x_pad,
                    attn_mask=attn_mask,
                    labels=labels,
                )
                loss = output.loss

                # embeds = projector(
                #     input_ids=input_ids,
                #     is_graph=is_graph,
                #     is_ts=is_ts,
                #     graph_x=graph_x,
                #     graph_x_pad=graph_x_pad,
                #     ts_x=ts_x,
                #     ts_x_pad=ts_x_pad,
                # )

                # output = language_model(
                #     inputs_embeds=embeds,
                #     attention_mask=attn_mask,
                #     labels=labels,
                #     use_cache=False,
                # )
                # loss = output.loss

                # loss = language_model(
                #     inputs_embeds=embeds,
                #     attention_mask=attn_mask,
                #     labels=labels,
                #     use_cache=False,
                # ).loss

            # loss.backward()
                accelerator.backward(loss)

                if config['max_grad_norm'] is not None and config['max_grad_norm'] > 0:
                    accelerator.clip_grad_norm_(model.parameters(), config['max_grad_norm'])

                optimizer.step()
                warmup_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            loss_detached = accelerator.gather_for_metrics(loss.detach()).mean().item()
            epoch_loss += loss_detached

            accelerator.log({"train/loss": loss_detached, "train/lr": warmup_scheduler.get_last_lr()[0]}, step=global_step)

            if accelerator.is_main_process and pbar is not None:
                if (step + 1) % config['log_interval'] == 0:
                    pbar.set_postfix({'loss': f"{loss_detached:.4f}"})
                pbar.update(1)

        if accelerator.is_main_process and pbar is not None:
            pbar.close()

        save_projector(accelerator, model, config['save_dir'], f"{config['model_save_name']}_epoch{epoch+1}_step{global_step}")

        epoch_avg_loss = epoch_loss / len(train_loader)
        accelerator.print(f"[Epoch {epoch+1}] Average Loss: {epoch_avg_loss:.4f}")

    save_projector(accelerator, model, config['save_dir'], f"{config['model_save_name']}_final_step{global_step}")

    accelerator.end_training()


if __name__ == "__main__":
    # export PYTHONPATH="/home/bwyin/project/Agent/MultiviewLLM
    # accelerate launch /home/bwyin/project/Agent/MultiviewLLM/src/trainer/MultiviewLLM/Instruction/train_instruction.py

    # Define configurations
    # from src.config.MultiviewLLM.Instruction.config import train_match_config as config
    #
    # main(config)

    # Define configurations
    from src.config.MultiviewLLM.Instruction.config import train_delinquency_config as config

    main(config)
