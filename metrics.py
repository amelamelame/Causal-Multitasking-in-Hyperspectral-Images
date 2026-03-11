"""
evaluation/metrics.py — Classification + Regression metrics for TAIGA.
"""
import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, accuracy_score, confusion_matrix)
from data.envi_reader import CAT_VARIABLES, REG_VARIABLES


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cls = {v: {"preds":[],"labels":[],"probs":[]} for v in CAT_VARIABLES}
        self.reg = {v: {"preds":[],"labels":[]} for v in REG_VARIABLES}

    def update(self, outputs, batch):
        for v in CAT_VARIABLES:
            logits = outputs["cls"][v]      # [B,C,H,W]
            lbl    = batch["cat"][v]        # [B]
            B,C,H,W = logits.shape
            c_logits = logits[:,:,H//2,W//2]  # center pixel
            probs  = c_logits.softmax(1).detach().cpu().numpy()
            preds  = c_logits.argmax(1).detach().cpu().numpy()
            labs   = lbl.detach().cpu().numpy()
            self.cls[v]["preds"].append(preds)
            self.cls[v]["labels"].append(labs)
            self.cls[v]["probs"].append(probs)

        for v in REG_VARIABLES:
            pred = outputs["reg"][v]        # [B,1,H,W]
            lbl  = batch["reg"][v]          # [B]
            H,W  = pred.shape[2:]
            c    = pred[:,0,H//2,W//2].detach().cpu().numpy()
            self.reg[v]["preds"].append(c)
            self.reg[v]["labels"].append(lbl.detach().cpu().numpy())

    def compute(self) -> dict:
        results = {}
        for v, n in CAT_VARIABLES.items():
            p  = np.concatenate(self.cls[v]["preds"])
            t  = np.concatenate(self.cls[v]["labels"])
            pr = np.concatenate(self.cls[v]["probs"])
            results[f"{v}/OA"]        = accuracy_score(t,p)*100
            results[f"{v}/MCA"]       = np.mean(confusion_matrix(t,p).diagonal()
                                         / (confusion_matrix(t,p).sum(1)+1e-8))*100
            results[f"{v}/Precision"] = precision_score(t,p,average="macro",zero_division=0)*100
            results[f"{v}/Recall"]    = recall_score(t,p,average="macro",zero_division=0)*100
            results[f"{v}/F1"]        = f1_score(t,p,average="macro",zero_division=0)*100
            try:
                results[f"{v}/AUC"] = roc_auc_score(t,pr,multi_class='ovr',average='macro')*100
            except: results[f"{v}/AUC"] = 0.

        rmse_all = []
        for v in REG_VARIABLES:
            p = np.concatenate(self.reg[v]["preds"])
            t = np.concatenate(self.reg[v]["labels"])
            rmse = float(np.sqrt(np.mean((p-t)**2)))
            mae  = float(np.mean(np.abs(p-t)))
            ss   = np.sum((t - t.mean())**2) + 1e-8
            r2   = float(1 - np.sum((p-t)**2)/ss)
            results[f"{v}/RMSE"] = rmse
            results[f"{v}/MAE"]  = mae
            results[f"{v}/R2"]   = r2
            rmse_all.append(rmse)

        results["mean_OA"]   = np.mean([results[f"{v}/OA"]  for v in CAT_VARIABLES])
        results["mean_RMSE"] = np.mean(rmse_all)
        return results

    def summary(self, r: dict) -> str:
        lines = ["\n" + "─"*65]
        lines.append(f"  {'CLASSIFICATION':}")
        for v in CAT_VARIABLES:
            lines.append(f"  {v:<30} OA={r[f'{v}/OA']:.2f}%  F1={r[f'{v}/F1']:.2f}%")
        lines.append(f"  {'REGRESSION':}")
        for v in REG_VARIABLES:
            lines.append(f"  {v:<30} RMSE={r[f'{v}/RMSE']:.4f}  R²={r[f'{v}/R2']:.4f}")
        lines.append(f"\n  Mean OA:   {r['mean_OA']:.2f}%")
        lines.append(f"  Mean RMSE: {r['mean_RMSE']:.4f}")
        lines.append("─"*65)
        return "\n".join(lines)
