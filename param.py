import argparse


def parameter_parser():
    """
    Parse command line arguments for the model.
    """
    parser = argparse.ArgumentParser(description="Run HeterogenousGraphCLAMIR.")

    # Dataset parameters
    parser.add_argument('--dataset',
                        type=str,
                        default='v2.0_495m383D',
                        help='Dataset name: v2.0_495m383D')

    # 添加数据路径参数 - 修复 AttributeError
    parser.add_argument('--data_path',
                        type=str,
                        default='./',
                        help='Path to data directory.')

    # Model architecture parameters
    parser.add_argument('--epoch',
                        type=int,
                        default=650,
                        help='Number of training epochs.')

    parser.add_argument('--nlayer',
                        type=int,
                        default=2,
                        help='Number of HGT layers.')

    parser.add_argument('--n_head',
                        type=int,
                        default=8,
                        help='Number of attention heads in HGT.')

    # Loss function parameters
    parser.add_argument('--alpha',
                        type=float,
                        default=0.5,
                        help='Weight for negative samples in loss function.')

    # Training parameters
    parser.add_argument('--validation',
                        type=int,
                        default=5,
                        help='Number of cross-validation folds.')

    parser.add_argument('--dropout',
                        type=float,
                        default=0.3,
                        help='Dropout rate.')

    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='Learning rate.')

    # Data dimensions (will be set automatically)
    parser.add_argument('--mi_num',
                        type=int,
                        default=495,
                        help='Number of miRNAs.')

    parser.add_argument('--dis_num',
                        type=int,
                        default=383,
                        help='Number of diseases.')

    # Hypergraph construction parameters
    parser.add_argument('--K_neigs',
                        nargs='+',
                        type=int,
                        default=[13],
                        help='K nearest neighbors for KNN hypergraph.')

    parser.add_argument('--clusters',
                        nargs='+',
                        type=int,
                        default=[9],
                        help='Number of clusters for K-means hypergraph.')

    # Contrastive learning parameters
    parser.add_argument('--cl_temperature',
                        type=float,
                        default=0.5,
                        help='Temperature for contrastive loss.')

    parser.add_argument('--cl_weight',
                        type=float,
                        default=1.0,
                        help='Weight for contrastive loss.')

    # Multi-type association parameters
    parser.add_argument('--enable_type_prediction',
                        type=bool,
                        default=True,
                        help='Enable multi-type association prediction.')

    parser.add_argument('--num_association_types',
                        type=int,
                        default=4,
                        help='Number of biological association types (4: circulation, epigenetics, target, genetics).')

    # [NEW] Cross-modal inter-view contrastive learning parameters
    parser.add_argument('--enable_inter_view_cl',
                        type=bool,
                        default=True,
                        help='Enable inter-view contrastive learning between miRNA and Disease.')

    parser.add_argument('--inter_view_weight',
                        type=float,
                        default=0.3,
                        help='Weight for inter-view contrastive loss.')

    parser.add_argument('--inter_view_temperature',
                        type=float,
                        default=0.5,
                        help='Temperature for inter-view contrastive loss.')

    parser.add_argument('--inter_view_margin',
                        type=float,
                        default=0.5,
                        help='Margin for inter-view contrastive loss.')

    # Optimization parameters
    parser.add_argument('--use_scheduler',
                        type=bool,
                        default=True,
                        help='Use learning rate scheduler.')

    parser.add_argument('--weight_decay',
                        type=float,
                        default=5e-5,  # ✅ 适中的正则化，防止过度惩罚
                        help='Weight decay for optimizer.')

    # Device parameters
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help='Device to use: cuda or cpu.')

    # Reproducibility
    parser.add_argument('--seed',
                        type=int,
                        default=1234,
                        help='Random seed.')

    # Evaluation metric
    parser.add_argument('--eval_metric',
                        type=str,
                        default='AUPR',
                        choices=['AUC', 'AUPR', 'F1'],
                        help='Primary evaluation metric.')

    # Dynamic hypergraph update
    parser.add_argument('--update_graph_frequency',
                        type=int,
                        default=50,
                        help='Frequency of hypergraph updates (in epochs).')

    parser.add_argument('--similarity_threshold',
                        type=float,
                        default=0.5,
                        help='Threshold for adaptive similarity in hypergraph.')

    # Advanced loss options
    parser.add_argument('--use_focal_loss',
                        type=bool,
                        default=True,
                        help='Use focal loss for handling class imbalance.')

    parser.add_argument('--focal_gamma',
                        type=float,
                        default=2.5,  # ✅ 优化后的gamma值，平衡难易样本
                        help='Gamma parameter for focal loss.')

    parser.add_argument('--class_weights',
                        nargs='+',
                        type=float,
                        default=None,
                        help='Class weights for loss function (4 values for 4 association types).')

    # Verbose output
    parser.add_argument('--verbose',
                        type=bool,
                        default=True,
                        help='Print detailed information.')

    # Ablation mode for paper Fig. 4 reproduction (additive — default 'none' preserves original behavior)
    parser.add_argument('--ablation',
                        type=str,
                        default='none',
                        choices=['none', 'no_cl', 'no_hgcn', 'no_avf', 'no_hgt', 'no_dv'],
                        help='Ablation variant: no_cl|no_hgcn|no_avf|no_hgt|no_dv (Fig.4 paper).')

    # Save paths
    parser.add_argument('--save_dir',
                        type=str,
                        default='./results/',
                        help='Directory for saving results.')

    parser.add_argument('--log_dir',
                        type=str,
                        default='./logs/',
                        help='Directory for logs.')

    args = parser.parse_args()

    # Validate and adjust parameters
    validate_and_adjust_parameters(args)

    return args


def validate_and_adjust_parameters(args):
    """
    Validate and adjust parameters to ensure consistency.
    """
    import torch

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA not available. Switching to CPU.")
        args.device = 'cpu'

    # Ensure hidden dimensions are compatible with attention heads
    if args.n_head > 0:
        # Hidden dimension should be divisible by number of heads
        base_hidden = 256
        if base_hidden % args.n_head != 0:
            adjusted_hidden = (base_hidden // args.n_head) * args.n_head
            if adjusted_hidden == 0:
                adjusted_hidden = args.n_head * 5
            print(f"[INFO] Adjusted hidden dimension to {adjusted_hidden} to match {args.n_head} attention heads")

    # Validate epoch number
    if args.epoch < 100:
        print("[WARNING] Number of epochs is very low. Consider using at least 100 epochs.")

    # Validate learning rate
    if args.lr > 0.01:
        print("[WARNING] Learning rate is high. Consider using a smaller value (e.g., 0.0001).")

    # Validate dropout
    if args.dropout > 0.5:
        print("[WARNING] Dropout rate is high. This may lead to underfitting.")

    # Validate alpha
    if args.alpha < 0 or args.alpha > 1:
        raise ValueError("Alpha must be between 0 and 1.")

    # Set class weights if not provided
    if args.class_weights is None:
        # 使用反比例权重，强调少数类以防止类别崩溃
        # Circulation: 367 (24.5%), Epigenetics: 157 (10.5%), Target: 293 (19.6%), Genetics: 681 (45.5%)
        counts = [367, 157, 293, 681]  # 实际样本数
        total = sum(counts)

        # 计算反比例权重（样本越少，权重越高）
        args.class_weights = [total / count for count in counts]

        # 归一化权重，使平均值为1.0
        avg_weight = sum(args.class_weights) / len(args.class_weights)
        args.class_weights = [w / avg_weight for w in args.class_weights]

        print("[INFO] Auto-computed BALANCED class weights (to prevent class collapse):")
        print(f"  Circulation  (367 samples): {args.class_weights[0]:.3f}")
        print(f"  Epigenetics  (157 samples): {args.class_weights[1]:.3f}")
        print(f"  Target       (293 samples): {args.class_weights[2]:.3f}")
        print(f"  Genetics     (681 samples): {args.class_weights[3]:.3f}")
        print(f"  [NOTE] Higher weights for minority classes to prevent model from only predicting majority class")

    # Validate number of association types
    if args.num_association_types != 4:
        print(f"[WARNING] num_association_types is set to {args.num_association_types}.")
        print("  The model expects 4 types: circulation, epigenetics, target, genetics.")

    # Validate inter-view CL parameters
    if args.enable_inter_view_cl:
        if args.inter_view_weight < 0 or args.inter_view_weight > 1:
            print("[WARNING] inter_view_weight should be between 0 and 1. Adjusting to 0.3")
            args.inter_view_weight = 0.3

    # Dataset-specific settings
    if args.dataset == 'v2.0_495m383D':
        args.mi_num = 495
        args.dis_num = 383
        print(f"[INFO] Using v2.0_495m383D dataset:")
        print(f"  miRNAs: {args.mi_num}")
        print(f"  Diseases: {args.dis_num}")
        print(f"  Expected associations: ~1498")
        print(f"  Types: 4 (Circulation, Epigenetics, Target, Genetics)")
    else:
        print(f"[WARNING] Unknown dataset: {args.dataset}")
        print("  Using default dimensions (495 miRNAs, 383 diseases)")

    # Validate K_neigs and clusters
    if not isinstance(args.K_neigs, list):
        args.K_neigs = [args.K_neigs]

    if not isinstance(args.clusters, list):
        args.clusters = [args.clusters]

    # Print configuration summary
    if args.verbose:
        print("\n" + "=" * 80)
        print("Configuration Summary")
        print("=" * 80)
        print(f"Dataset: {args.dataset}")
        print(f"Data Path: {args.data_path}")
        print(f"Model: HeterogenousGraphCLAMIR")
        print(f"  - HGT Layers: {args.nlayer}")
        print(f"  - Attention Heads: {args.n_head}")
        print(f"  - Association Types: {args.num_association_types}")
        print(f"  - Inter-view CL: {'Enabled' if args.enable_inter_view_cl else 'Disabled'}")
        if args.enable_inter_view_cl:
            print(f"    - Weight: {args.inter_view_weight}")
            print(f"    - Temperature: {args.inter_view_temperature}")
            print(f"    - Margin: {args.inter_view_margin}")
        print(f"\nTraining:")
        print(f"  - Epochs: {args.epoch}")
        print(f"  - Learning Rate: {args.lr}")
        print(f"  - Dropout: {args.dropout}")
        print(f"  - Weight Decay: {args.weight_decay}")
        print(f"  - Focal Loss: {'Enabled' if args.use_focal_loss else 'Disabled'}")
        print(f"\nEvaluation:")
        print(f"  - Cross-Validation Folds: {args.validation}")
        print(f"  - Primary Metric: {args.eval_metric}")
        print(f"\nDevice: {args.device}")
        print(f"Random Seed: {args.seed}")
        print("=" * 80 + "\n")

    return args


def print_args(args):
    """
    Print all arguments in a formatted way.
    """
    print("\n" + "=" * 80)
    print("Hyperparameters")
    print("=" * 80)

    # Group parameters by category
    categories = {
        'Dataset': ['dataset', 'data_path', 'mi_num', 'dis_num'],
        'Model Architecture': ['nlayer', 'n_head', 'dropout', 'num_association_types'],
        'Training': ['epoch', 'lr', 'alpha', 'weight_decay', 'use_scheduler'],
        'Contrastive Learning': ['cl_temperature', 'cl_weight', 'enable_inter_view_cl',
                                 'inter_view_weight', 'inter_view_temperature', 'inter_view_margin'],
        'Hypergraph': ['K_neigs', 'clusters', 'update_graph_frequency', 'similarity_threshold'],
        'Loss Function': ['use_focal_loss', 'focal_gamma', 'class_weights'],
        'Evaluation': ['validation', 'eval_metric'],
        'System': ['device', 'seed', 'verbose']
    }

    for category, params in categories.items():
        print(f"\n{category}:")
        for param in params:
            if hasattr(args, param):
                value = getattr(args, param)
                if isinstance(value, list) and len(value) > 5:
                    value = f"[{value[0]}, ..., {value[-1]}] (length: {len(value)})"
                print(f"  {param:30s}: {value}")

    print("=" * 80 + "\n")


# Example usage
if __name__ == '__main__':
    args = parameter_parser()
    print_args(args)

    # Test parameter validation
    print("\n[TEST] Parameter validation passed successfully!")
    print(f"[TEST] Model will use {args.device}")
    print(f"[TEST] Training for {args.epoch} epochs")
    print(f"[TEST] Using {args.validation}-fold cross-validation")
    print(f"[TEST] Inter-view CL: {'Enabled' if args.enable_inter_view_cl else 'Disabled'}")