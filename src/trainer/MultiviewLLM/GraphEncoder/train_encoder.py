'''
Trainer for graph encoder using contrastive learning.

Inputs:
    - Configuration dictionary containing training parameters.

Outputs:
    - Trained graph encoder model saved to specified path.
'''

import json
import torch
import wandb
# from torch.utils.tensorboard import SummaryWriter   # Use wandb instead
from torch_geometric.data import DataLoader
from src.dataset.MultiviewLLM.GraphEncoder.dataset import make_dataset
from src.model.MultiviewLLM.GraphEncoder.model import EncoderLearner
from src.utils.seed_everything import seed_everything
import GCL.losses as L
import GCL.augmentors as A
from torch.optim import Adam
from GCL.eval import get_split
from src.utils.MultiviewLLM.GraphEncoder.evaluator import SVMEvaluator
from GCL.models import DualBranchContrast


def train(encoder_model, contrast_model, dataloader, optimizer, writer, epoch, device, semantic_contrast_model=None):
    '''
    Training function for one epoch.

    Args:
        encoder_model (torch.nn.Module): The graph encoder model.
        contrast_model (torch.nn.Module): The contrastive learning model.
        dataloader (DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        writer (SummaryWriter or wandb): Writer for logging training metrics.
        epoch (int): Current epoch number.
        device (torch.device): Device to run the training on.
        semantic_contrast_model (torch.nn.Module, optional): Semantic contrastive model if used.
    '''
    encoder_model.train()
    epoch_loss = 0
    for train_step, data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()

        z, _, _, _, g1, g2, s = encoder_model(data.x, data.edge_index, data.edge_attr, data.batch)
        g1, g2 = [encoder_model.project_graph(g) for g in [g1, g2]]
        main_loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        if semantic_contrast_model:
            z = encoder_model.project_node(z)
            s = encoder_model.project_mcc(s)
            semantic_loss = semantic_contrast_model(h1=z, h2=s, batch=data.batch)
        else:
            semantic_loss = 0

        loss = main_loss + semantic_loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # print(f'{train_step}/{len(dataloader)}: Loss={loss.item():.4f}')
        # step = epoch * len(dataloader) + train_step
        # writer.add_scalar('Train/Main_Loss', main_loss.item(), step)
        # writer.add_scalar('Train/Semantic_Loss', semantic_loss.item() if semantic_contrast_model else 0, step)
        # writer.add_scalar('Train/Loss', loss.item(), step)
        wandb.log({'Train/Main_Loss': main_loss.item(),
                   'Train/Semantic_Loss': semantic_loss.item() if semantic_contrast_model else 0,
                   'Train/Loss': loss.item(),
                   'epoch': epoch})

        torch.cuda.empty_cache()
    return epoch_loss


def test(encoder_model, dataloader):
    '''
    Evaluation function using SVM on the learned graph representations.

    Args:
        encoder_model (torch.nn.Module): The trained graph encoder model.
        dataloader (DataLoader): DataLoader for the evaluation dataset.

    Returns:
        dict: Dictionary containing evaluation metrics (micro_f1, macro_f1).
    '''
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')

        _, g, _, _, _, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr, data.batch)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = SVMEvaluator(linear=True).evaluate(x, y, split)
    return result


def main(config):
    '''
    Main function to train and evaluate the graph encoder.

    Args:
        config (dict): Configuration dictionary containing training parameters.
    '''
    # Set device and writer
    seed_everything(config['seed'])
    device = torch.device(config['device'])
    # To use tensorboard, run in terminal:
    # tensorboard --logdir=/data/bwyin/project/MultiviewLLM/tensorboard_logs/MultiviewLLM/GraphEncoder --bind_all
    # writer = SummaryWriter(log_dir=config['writer_path'])
    run = wandb.init(project=config['project'], entity=config['entity'], config=config)

    # Load dataset
    dataset = make_dataset(config['dataset_path'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)

    # Load encoder and determine input dimension
    with open(config['encoder_path'], 'r') as f:
        encoder = json.load(f)
    mcc_num = len(encoder['mcc_cde_encoder'])
    edge_dim = dataset[0].edge_attr.size(1) if dataset[0].edge_attr is not None else None
    edge_dim = edge_dim if config['use_edge_attr'] else None

    # load mcc embeddings and enhance node features if specified
    mcc_embed = torch.load(config['mcc_embed_path']).to(device)

    # Define view functions
    aug1 = A.Identity()
    aug2 = A.RandomChoice([A.NodeDropping(pn=config['node_drop_prob']),
                           A.FeatureMasking(pf=config['feature_mask_prob']),
                           A.EdgeRemoving(pe=config['edge_drop_prob'])], 1)
    # Due to unknown bug, we do not use RWSampling currently
    # aug2 = A.RandomChoice([A.RWSampling(num_seeds=config['rws_start_size'], walk_length=config['rws_walk_length']),
    #                        A.NodeDropping(pn=config['node_drop_prob']),
    #                        A.FeatureMasking(pf=config['feature_mask_prob']),
    #                        A.EdgeRemoving(pe=config['edge_drop_prob'])], 1)

    # Define models and optimizer
    encoder_model = EncoderLearner(mcc_num=mcc_num,
                                   edge_dim=edge_dim,
                                   layer_mode=config['layer_mode'],
                                   hidden_dim=config['hidden_dim'],
                                   num_layers=config['num_layers'],
                                   augmentor=(aug1, aug2),
                                   mcc_embed=mcc_embed,
                                   semantic_initial=config['semantic_initial']).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
    if config['semantic_contrastive']:
        semantic_contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L').to(device)
    else:
        semantic_contrast_model = None
    optimizer = Adam(encoder_model.parameters(), lr=config['learning_rate'])

    # Training and evaluation functions
    test_result = test(encoder_model, dataloader)
    for k, v in test_result.items():
        run.summary[f'Initial/{k}'] = v
    for epoch in range(config['epochs']):
        # train(encoder_model, contrast_model, dataloader, optimizer, writer, epoch, device, semantic_contrast_model)  # using tensorboard
        train(encoder_model, contrast_model, dataloader, optimizer, run, epoch, device, semantic_contrast_model)  # using wandb
    test_result = test(encoder_model, dataloader)
    for k, v in test_result.items():
        run.summary[f'Final/{k}'] = v

    # Save the trained model
    model_save_path = config['model_save_path'] / config['dataset_path'].name.replace('_graph.pt', '_model.pth')
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder_model.state_dict(), model_save_path)

    # Save the embeddings
    encoder_model.eval()
    all_node_embeds = {}
    all_graph_embeds = {}
    for data in dataloader:
        data = data.to(device)
        z, g, _, _, _, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr, data.batch)
        sizes = torch.bincount(data.batch)
        z_pre_graph = list(torch.split(z, sizes.tolist(), dim=0))
        indexes = data['meta_info']['index'].tolist()
        for i, idx in enumerate(indexes):
            all_node_embeds[idx] = z_pre_graph[i].cpu()
            all_graph_embeds[idx] = g[i].unsqueeze(0).cpu()

    graph_embed_save_path = config['embed_save_path'] / config['dataset_path'].name.replace('_graph.pt', '_graph_embed.pt')
    with open(graph_embed_save_path, 'wb') as f:
        torch.save(all_graph_embeds, f)
    node_embed_save_path = config['embed_save_path'] / config['dataset_path'].name.replace('_graph.pt', '_node_embed.pt')
    with open(node_embed_save_path, 'wb') as f:
        torch.save(all_node_embeds, f)

    # writer.close()
    run.finish()


if __name__ == '__main__':
    # Define the configuration
    from src.config.MultiviewLLM.GraphEncoder.config import train_config as config

    main(config)

    # # Run the training and evaluation
    # for layer in ['GINE', 'SAGE']:
    #     config['layer_mode'] = layer
    #     for use_edge in [True, False]:
    #         config['use_edge_attr'] = use_edge
    #         for semantic_initial in [True, False]:
    #             config['semantic_initial'] = semantic_initial
    #             for semantic_contrastive in [True, False]:
    #                 config['semantic_contrastive'] = semantic_contrastive
    #                 main(config)
