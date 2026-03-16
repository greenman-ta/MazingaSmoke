import os


def apply_finetune_lr(optimizer):

    optimizer.param_groups[0]["lr"] = 1e-5   
    optimizer.param_groups[1]["lr"] = 1e-4   
    optimizer.param_groups[2]["lr"] = 1e-5   

def apply_finetune_freeze(model):
    # freeze primissimi layer backbone
    frozen = 0
    for name, p in model.backbone.named_parameters():
        if any(name.startswith(pref) for pref in ["backbone.blocks_2", "backbone.blocks_3", "backbone.blocks_4"]):

            p.requires_grad = False
            frozen += 1

    print(f"[-GhostV2policy] frozen n: {frozen}")


def resolve_resume_policy(args, last_ckpt_path, finetune_init_weights_path):
    """
    Ritorna dict:
      action: 'resume_full' | 'init_weights_only' | 'fresh'
      ckpt_path: path o None
    """
    if args.finetune:
        if os.path.isfile(last_ckpt_path):
            return {"action": "resume_full", "ckpt_path": last_ckpt_path}
        else:
            return {"action": "init_weights_only", "ckpt_path": finetune_init_weights_path}
    else:
        if args.resume_checkpoint:
            return {"action": "resume_full", "ckpt_path": args.resume_checkpoint}
        else:
            return {"action": "fresh", "ckpt_path": None}