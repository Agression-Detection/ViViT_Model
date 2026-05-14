from transformers import VivitForVideoClassification
from torch.nn.parallel import DistributedDataParallel as DDP

def get_model(device, is_dist, local_rank):
    model = VivitForVideoClassification.from_pretrained(
        "google/vivit-b-16x2-kinetics400",
        num_labels=2,
        num_frames=10,
        ignore_mismatched_sizes=True,
        use_safetensors=True,
    )
    model.config.id2label = {0: "non_violent", 1: "violent"}
    model.config.label2id = {"non_violent": 0, "violent": 1}
    for param in model.vivit.parameters():
        param.requires_grad = False

    model = model.to(device)
    if is_dist: model = DDP(model, device_ids=[local_rank])
    return model