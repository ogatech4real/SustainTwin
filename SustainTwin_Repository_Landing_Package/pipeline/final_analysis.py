from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix

from .config import WP1_DIR, WP2_DIR, WP3_DIR, OUTPUT_ROOT
from .utils import ensure_dirs, save_json

FINAL_DIR = OUTPUT_ROOT / "final_analysis"
FIG_DIR = FINAL_DIR / "figures"
TAB_DIR = FINAL_DIR / "tables"

# Color-blind-aware Okabe-Ito palette; fixed to keep figures consistent across the paper.
C = {"blue":"#0072B2", "orange":"#E69F00", "green":"#009E73", "red":"#D55E00",
     "purple":"#CC79A7", "sky":"#56B4E9", "black":"#222222", "grey":"#7A7A7A"}
FAULT_COLOR = {"F1":C["blue"], "F2":C["orange"], "F3":C["green"], "F4":C["red"], "F0":C["grey"]}
LABEL = {"temperature_c":"Temperature (°C)", "actual_rpm":"Speed (rpm)", "current_a":"Current (A)",
         "vibration_g":"Vibration (g)", "power_w":"Power (W)", "cooling_rate_c_per_s":"Cooling rate (°C s⁻¹)",
         "cooling_efficiency":"Cooling efficiency (-)", "thermal_resistance_c_per_w":"Thermal resistance (°C W⁻¹)",
         "effective_process_heat_load_w":"Effective heat load (W)"}

def _style():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.titlesize":10, "axes.labelsize":9,
                         "legend.fontsize":8, "xtick.labelsize":8, "ytick.labelsize":8, "axes.linewidth":0.8,
                         "lines.linewidth":1.8, "savefig.transparent":False})

def _save(fig, stem):
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

def _severity_figure(master):
    steady = master[(master.experiment_phase == "FAULT_ACTIVE") & master.experiment_elapsed_s.between(120,480)]
    specs=[("F1","vibration_g"),("F2","current_a"),("F3","temperature_c"),("F4","temperature_c")]
    fig, axes = plt.subplots(2,2,figsize=(7.2,5.5))
    for ax,(fault,signal) in zip(axes.flat,specs):
        g=steady[steady.selected_fault_code==fault].groupby("selected_fault_severity")[signal].agg(["mean","std"]).sort_index()
        ax.errorbar(g.index,g["mean"],yerr=g["std"].fillna(0),marker="o",capsize=3,color=FAULT_COLOR[fault])
        ax.set_title(f"{fault} — { {'F1':'Mechanical imbalance','F2':'Electrical overcurrent','F3':'Cooling degradation','F4':'Thermal overload'}[fault] }")
        ax.set_xlabel("Fault severity"); ax.set_ylabel(LABEL[signal]); ax.set_xticks([0.3,0.6,0.9]); ax.grid(alpha=.18)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Controlled severity response of primary fault mechanisms", fontweight="bold", y=1.01)
    fig.tight_layout(); _save(fig,"Fig_01_severity_response")

def _temporal_figure(master):
    fig, axes=plt.subplots(4,1,figsize=(7.2,8.2),sharex=True)
    signal={"F1":"vibration_g","F2":"current_a","F3":"temperature_c","F4":"temperature_c"}
    for ax,fault in zip(axes,["F1","F2","F3","F4"]):
        g=master[(master.selected_fault_code==fault)&np.isclose(pd.to_numeric(master.selected_fault_severity),0.6)].sort_values("experiment_elapsed_s")
        ax.plot(g.experiment_elapsed_s,g[signal[fault]],color=FAULT_COLOR[fault])
        ax.axvspan(0,60,color=C["grey"],alpha=.08); ax.axvspan(60,540,color=FAULT_COLOR[fault],alpha=.07); ax.axvspan(540,600,color=C["grey"],alpha=.08)
        ax.axvline(60,color=C["black"],ls="--",lw=.8); ax.axvline(540,color=C["black"],ls="--",lw=.8)
        ax.set_ylabel(LABEL[signal[fault]]); ax.set_title(f"{fault}, severity 0.6",loc="left",fontweight="bold"); ax.grid(alpha=.15); ax.spines[["top","right"]].set_visible(False)
    axes[-1].set_xlabel("Experiment elapsed time (s)")
    fig.suptitle("Temporal fault onset, persistence and recovery",fontweight="bold",y=.995)
    fig.tight_layout(); _save(fig,"Fig_02_temporal_fault_response")

def _external_figure():
    current=pd.read_csv(WP2_DIR/"current_secondary_validation.csv")
    em=pd.read_csv(WP2_DIR/"electromechanical_validation.csv")
    rows=[]
    for df in [current,em]:
        for _,r in df.iterrows():
            if pd.notna(r.twin_pearson_r) and pd.notna(r.real_pearson_r):
                rows.append({"relationship":f"{r.variable_1.title()}–{r.variable_2.title()}","SustainTwin":r.twin_pearson_r,"Industrial":r.real_pearson_r})
    d=pd.DataFrame(rows).drop_duplicates("relationship")
    d.to_csv(TAB_DIR/"table_external_comparable_correlations.csv",index=False)
    fig,ax=plt.subplots(figsize=(7.2,3.7)); x=np.arange(len(d)); w=.34
    ax.bar(x-w/2,d.SustainTwin,w,label="SustainTwin F0",color=C["blue"]); ax.bar(x+w/2,d.Industrial,w,label="Industrial fan",color=C["orange"])
    ax.set_xticks(x,d.relationship); ax.set_ylabel("Pearson correlation, r"); ax.set_ylim(0,1.08); ax.legend(frameon=False,ncol=2)
    ax.grid(axis="y",alpha=.18); ax.spines[["top","right"]].set_visible(False)
    ax.set_title("Empirically comparable electromechanical relationships",fontweight="bold")
    fig.tight_layout(); _save(fig,"Fig_03_external_behavioural_validation")

def _ablation_figure():
    d=pd.read_csv(WP3_DIR/"physics_feature_ablation.csv")
    order=["LogisticRegression","HistGradientBoosting","RandomForest"]
    d["model"]=pd.Categorical(d.model,categories=order,ordered=True); d=d.sort_values("model")
    d.to_csv(TAB_DIR/"table_physics_feature_ablation.csv",index=False)
    fig,ax=plt.subplots(figsize=(7.2,3.8)); x=np.arange(len(d)); w=.34
    ax.bar(x-w/2,d.observable,w,label="Observable",color=C["grey"]); ax.bar(x+w/2,d.physics_augmented,w,label="Physics-augmented",color=C["blue"])
    for i,r in enumerate(d.itertuples()): ax.text(i+w/2,r.physics_augmented+.008,f"+{r.oof_macro_f1_gain_physics:.3f}",ha="center",fontsize=8,fontweight="bold")
    ax.set_xticks(x,["Logistic regression","Hist. gradient boosting","Random forest"]); ax.set_ylabel("OOF macro-F1"); ax.set_ylim(0,1.02)
    ax.legend(frameon=False,ncol=2); ax.grid(axis="y",alpha=.18); ax.spines[["top","right"]].set_visible(False)
    ax.set_title("Physics-derived features improve continuous multiclass diagnosis",fontweight="bold")
    fig.tight_layout(); _save(fig,"Fig_04_physics_feature_ablation")

def _task_figure():
    m=pd.read_csv(WP3_DIR/"model_comparison_metrics.csv")
    rows=[]
    for task,label in [("full_temporal_multiclass","Full temporal F0–F4"),("fault_active_multiclass","Fault-active F1–F4"),("binary_healthy_fault","Healthy vs fault")]:
        g=m[m.task==task].sort_values("oof_macro_f1_expected_labels",ascending=False).iloc[0]
        rows.append({"task":label,"dataset":g.dataset,"model":g.model,"accuracy":g.oof_accuracy,"balanced_accuracy":g.oof_balanced_accuracy,"macro_f1":g.oof_macro_f1_expected_labels})
    d=pd.DataFrame(rows); d.to_csv(TAB_DIR/"table_headline_ai_results.csv",index=False)
    fig,ax=plt.subplots(figsize=(7.2,3.7)); x=np.arange(len(d)); w=.25
    ax.bar(x-w,d.accuracy,w,label="Accuracy",color=C["sky"]); ax.bar(x,d.balanced_accuracy,w,label="Balanced accuracy",color=C["green"]); ax.bar(x+w,d.macro_f1,w,label="Macro-F1",color=C["blue"])
    ax.set_xticks(x,d.task); ax.set_ylim(0,1.05); ax.set_ylabel("Score"); ax.legend(frameon=False,ncol=3); ax.grid(axis="y",alpha=.18); ax.spines[["top","right"]].set_visible(False)
    ax.set_title("Run-grouped out-of-fold diagnostic performance",fontweight="bold")
    fig.tight_layout(); _save(fig,"Fig_05_grouped_ai_performance")

def _confusion_figure():
    p=pd.read_csv(WP3_DIR/"grouped_cv_predictions.csv")
    # Best full-temporal frozen model
    g=p[(p.task=="full_temporal_multiclass")&(p.dataset=="physics_augmented")&(p.model=="LogisticRegression")]
    labels=["F0","F1","F2","F3","F4"]; cm=confusion_matrix(g["true"].astype(str),g["pred"].astype(str),labels=labels)
    norm=cm/cm.sum(axis=1,keepdims=True)
    pd.DataFrame(cm,index=labels,columns=labels).to_csv(TAB_DIR/"table_confusion_full_temporal_counts.csv")
    fig,ax=plt.subplots(figsize=(5.4,4.6)); im=ax.imshow(norm,vmin=0,vmax=1,cmap="Blues")
    for i in range(5):
        for j in range(5):
            ax.text(j,i,f"{norm[i,j]*100:.1f}%\n(n={cm[i,j]})",ha="center",va="center",fontsize=7,color="white" if norm[i,j]>.55 else C["black"])
    ax.set_xticks(range(5),labels); ax.set_yticks(range(5),labels); ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title("Full-temporal confusion matrix\nPhysics-augmented logistic regression",fontweight="bold")
    cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.04); cb.set_label("Row-normalized proportion")
    fig.tight_layout(); _save(fig,"Fig_06_full_temporal_confusion_matrix")

def _severity_transfer_figure():
    d=pd.read_csv(WP3_DIR/"leave_one_severity_out_fault_active.csv")
    # retain selected/best model rows where available; summarize max per dataset/severity to avoid duplicate candidates
    metric="macro_f1_expected_labels" if "macro_f1_expected_labels" in d.columns else "macro_f1"
    sevcol="held_out_severity" if "held_out_severity" in d.columns else "severity"
    s=d.groupby(["dataset",sevcol],as_index=False)[metric].max()
    s.to_csv(TAB_DIR/"table_leave_one_severity_out.csv",index=False)
    fig,ax=plt.subplots(figsize=(7.2,3.7))
    for dataset,color,label in [("observable",C["grey"],"Observable"),("physics_augmented",C["blue"],"Physics-augmented")]:
        g=s[s.dataset==dataset].sort_values(sevcol); ax.plot(g[sevcol],g[metric],marker="o",color=color,label=label)
    ax.set_xticks([.3,.6,.9]); ax.set_xlabel("Entirely withheld severity"); ax.set_ylabel("Macro-F1"); ax.set_ylim(0,1.05); ax.legend(frameon=False)
    ax.grid(alpha=.18); ax.spines[["top","right"]].set_visible(False); ax.set_title("Transfer across the controlled severity continuum",fontweight="bold")
    fig.tight_layout(); _save(fig,"Fig_07_leave_one_severity_out")

def run():
    ensure_dirs(FIG_DIR,TAB_DIR); _style()
    master=pd.read_csv(WP1_DIR/"experiments_master_clean.csv")
    master["experiment_elapsed_s"]=pd.to_numeric(master.experiment_elapsed_s,errors="coerce")
    master=master[master.run_inclusion_status=="PRIMARY"].copy()
    _severity_figure(master); _temporal_figure(master); _external_figure(); _ablation_figure(); _task_figure(); _confusion_figure(); _severity_transfer_figure()
    # Core manuscript tables copied/condensed into one location.
    pd.read_csv(WP1_DIR/"severity_response_summary.csv").to_csv(TAB_DIR/"table_severity_response.csv",index=False)
    pd.read_csv(WP1_DIR/"run_quality_audit.csv").to_csv(TAB_DIR/"table_run_quality.csv",index=False)
    manifest={"pipeline_version":"2.2-final","figure_count":7,"formats":["PNG 600 dpi","PDF vector"],"status":"FROZEN_FOR_MANUSCRIPT",
              "figure_directory":str(FIG_DIR),"table_directory":str(TAB_DIR)}
    save_json(manifest,FINAL_DIR/"final_analysis_manifest.json")
    return manifest

if __name__=="__main__": print(run())
