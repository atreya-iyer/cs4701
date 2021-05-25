from collections import defaultdict
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, precision_recall_curve
from matplotlib import pyplot as plt
modelnames = ["EfficientNetB4", "EfficientNetB4ST", "EfficientNetAutoAttB4", "EfficientNetAutoAttB4ST", "Xception", "FFPPEfficientNetB4", "FFPPEfficientNetB4ST", "FFPPEfficientNetAutoAttB4", "FFPPEfficientNetAutoAttB4ST", "FFPPXception"]

with open("Model Selection Data.txt", "r") as file:
    lines = file.readlines()
    lines = [lines[i][1:-2] for i in range(len(lines)) if i % 3 != 0]
    #print(lines)
    correct = [lines[i] for i in range(len(lines)) if i % 2 == 0]
    preds = [lines[i] for i in range(len(lines)) if i % 2 == 1]
    modelfp = {}
    modelfn = {}
    modeltp = {}
    modeltn = {}
    y = []
    modelprobs = defaultdict(list)
    for vid in range(len(correct)):
        currentright = correct[vid].split(', ')
        currentpred = preds[vid].split(', ')
        intcr = [int(i) for i in currentright]
        intp = [float(i) for i in currentpred]
        if intcr[0] == 1: #if model was correct
            if intp[0] < 0.5: #if it predicted deepfake
                y.append(1)
            else:
                y.append(0)
        else:
            if intp[0] < 0.5: #if it predicted deepfake
                y.append(0)
            else:
                y.append(1)
        for model in range(len(currentright)):
            right = int(currentright[model])
            prob = float(currentpred[model])
            modelprobs[model].append(1-prob)
            pred = round(prob)
            if right == 1:
                if pred == 1:
                    modeltn[model] = modeltn.get(model, 0) + 1
                else:
                    modeltp[model] = modeltp.get(model, 0) + 1
            else:
                if pred == 1:
                    modelfn[model] = modelfn.get(model, 0) + 1
                else:
                    modelfp[model] = modelfp.get(model, 0) + 1
    f1scores = []
    accs = []
    totals = []
    aucs = []
    for i in range(10):
        f1scores.append(modeltp.get(i, 0) / (modeltp.get(i, 0) + .5*(modelfp.get(i, 0) + modelfn.get(i, 0))))
        total = modeltp.get(i, 0) + modeltn.get(i, 0) + modelfn.get(i, 0) + modelfp.get(i, 0)
        totals.append(total)
        accs.append((modeltp.get(i, 0) + modeltn.get(i, 0))/ total)
        aucs.append(roc_auc_score(y, modelprobs[i]))
        fpr, tpr, _ = roc_curve(y, modelprobs[i])
        plt.plot(fpr, tpr, label=modelnames[i])
    fpr, tpr, _ = roc_curve(y, [.94]*144)
    plt.plot(fpr, tpr, label="Ours")
    plt.legend()
    plt.show()

    print(f1scores, totals, accs, aucs)
