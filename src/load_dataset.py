import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.data import Dataset

class loadDataset(Dataset):
    # Path fisso al file di patch con le nuove etichette/pesi
    PATCH_JSON = "./patch.json"

    def __init__(self, split_json, rgb_dir, num_segments=8, transform=None):
        import json
        self.rgb_dir = rgb_dir
        self.num_segments = num_segments
        self.transform = transform

        # Carica il metadata originale per questo split
        with open(split_json) as f:
            self.samples = json.load(f)

        # Carica le patch
        self.patch_by_orig_name = {}

        def _load_patch(patch_path, tag):
            if patch_path is None or (not os.path.isfile(patch_path)):
                print(f"[PATCH] Nessun file di patch trovato a {patch_path} ({tag} disattivata)")
                return []
            print(f"[PATCH] Carico {tag} da {patch_path}")
            with open(patch_path, "r") as pf:
                return json.load(pf)

        # ---- PATCH 1 ----
        patch_entries_1 = _load_patch(self.PATCH_JSON, "PATCH_1")
        n_total_1, n_used_1 = 0, 0
        for e in patch_entries_1:
            n_total_1 += 1
            orig_name = e.get("orig_file_name") or e.get("clip_name")
            if orig_name is None:
                continue

            new_label  = e.get("new_label", None)
            new_weight = e.get("new_weight", None)
            if new_label is None and new_weight is None:
                continue

            self.patch_by_orig_name[orig_name] = {"new_label": new_label, "new_weight": new_weight}
            n_used_1 += 1

        print(f"[PATCH] PATCH_1: Entry totali={n_total_1} | applicabili={n_used_1}")

        

    def __len__(self):
        return len(self.samples)

    def _sample_indices(self, T):
        if T <= self.num_segments:
            return list(range(T)) + [T-1]*(self.num_segments-T)
        step = T / self.num_segments
        return [min(T-1, int(step*(i+0.5))) for i in range(self.num_segments)]

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # nome originale (180x180) come nel metadata
        orig_file_name = sample["file_name"]
        # nome del file .npy 224x224
        name = orig_file_name.replace("-180-180-", "-224-224-")

        # label/weight originali
        label = int(sample["label"])
        weight = float(sample["weight"])

        # APPLICA PATCH
        patch = self.patch_by_orig_name.get(orig_file_name, None)
        if patch is not None:
            if patch.get("new_label") is not None:
                #print(f"[PATCH] Applicata patch su {orig_file_name}: {label} -> {patch['new_label']}")
                label = int(patch["new_label"])
            if patch.get("new_weight") is not None:
                weight = float(patch["new_weight"])

        # carica i frame
        rgb_sel = np.load(os.path.join(self.rgb_dir, f"{name}.npy"))      # [F,H,W,3] 
        F_tot = rgb_sel.shape[0]
        idx_sel = self._sample_indices(F_tot)      # len = self.num_segments
        rgb_sel = rgb_sel[idx_sel]

        if self.transform is not None:
            out = self.transform(rgb_sel)
            if isinstance(out, torch.Tensor) and out.ndim == 4:
                
                if out.shape[0] == 3 and out.shape[1] != 3:
                    rgb_tensor = out.permute(1, 0, 2, 3).contiguous()
                else:
                    rgb_tensor = out.contiguous()
            else:
                rgb_tensor = torch.from_numpy(rgb_sel).permute(0, 3, 1, 2).float() / 255.0
        else:
            rgb_tensor = torch.from_numpy(rgb_sel).permute(0, 3, 1, 2).float() / 255.0

        assert rgb_tensor.ndim == 4
        assert rgb_tensor.shape[1] == 3
        return {
            "rgb": rgb_tensor,
            "label": torch.tensor(label),
            "clip_name": name,
            "weight": torch.tensor(weight, dtype=torch.float32),
        }

    def get_label_counts(self):
        n_pos = 0
        n_neg = 0
        for sample in self.samples:
            orig_file_name = sample["file_name"]
            label = int(sample["label"])

            patch = self.patch_by_orig_name.get(orig_file_name, None)
            if patch is not None and patch.get("new_label") is not None:
                label = int(patch["new_label"])

            if label == 1:
                n_pos += 1
            else:
                n_neg += 1
        return n_pos, n_neg