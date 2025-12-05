from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from src.utils.MultiviewLLM.Instruction.utils_with_encoder import *
from src.utils.seed_everything import seed_everything


def save_model(accelerator: Accelerator, projector, save_dir: Path, name: str):
    """
    保存除llm以外的所有模型参数
    """
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(projector)
        ckpt_path = save_dir / f"{name}.pt"
        # 除了llm以外的所有参数
        graph_state_dict = unwrapped.graph_model.state_dict()
        ts_state_dict = unwrapped.ts_model.state_dict()
        projector_state_dict = unwrapped.projector.state_dict()
        torch.save({
            'graph_model': graph_state_dict,
            'ts_model': ts_state_dict,
            'projector': projector_state_dict,
        }, ckpt_path)
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
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
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

    # Prepare everything with accelerator
    model, optimizer, train_loader, warmup_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        warmup_scheduler,
    )

    accelerator.print(
        f"process_index={accelerator.process_index}, "
        f"num_processes={accelerator.num_processes}, "
        f"device={accelerator.device}"
    )

    # Prepare paths for saving models
    global_step = 0
    for epoch in range(config['num_epochs']):
        model.train()

        if accelerator.is_main_process:
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        else:
            pbar = None

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                output = model(batch)
                loss = output.loss

                accelerator.backward(loss)

                if config['max_grad_norm'] is not None and config['max_grad_norm'] > 0:
                    accelerator.clip_grad_norm_(model.parameters(), config['max_grad_norm'])

                optimizer.step()
                warmup_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

            if (step + 1) % config['log_interval'] == 0:
                loss_detached = accelerator.gather_for_metrics(loss.detach()).mean().item()
                accelerator.log({"train/loss": loss_detached, "train/lr": warmup_scheduler.get_last_lr()[0]}, step=global_step)
            if accelerator.is_main_process and pbar is not None:
                pbar.update(1)

        save_name = config['model_save_name'].format(graph_query_num=config['graph_query_num'], ts_query_num=config['ts_query_num'])+f"_epoch{epoch+1}_step{global_step}"
        save_model(accelerator, model, config['save_dir'], save_name)

    save_name = config['model_save_name'].format(graph_query_num=config['graph_query_num'], ts_query_num=config['ts_query_num'])+f"_final_step{global_step}"
    save_model(accelerator, model, config['save_dir'], save_name)

    accelerator.end_training()


if __name__ == "__main__":
    # export PYTHONPATH="/home/bwyin/project/Agent/MultiviewLLM"
    # accelerate launch /home/bwyin/project/Agent/MultiviewLLM/src/trainer/MultiviewLLM/Instruction/train_with_encoder.py

    # set gpu device = 1
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # Define configurations
    from src.config.MultiviewLLM.Instruction.config_with_encoder import train_config as config

    main(config)
