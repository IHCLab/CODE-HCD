function [OA, kappa, precision, recall] = evaluate(output, gt1D)

    con = confusionmat(gt1D, output);
    TP = con(2,2);
    FP = con(1,2);
    FN = con(2,1);
    TN = con(1,1);
    OA = (TP+TN)/(TP+TN+FP+FN);
    PRE = ((TP+FP)*(TP+FN)+(TN+FP)*(TN+FN))/(TP+TN+FP+FN)^2;
    kappa = (OA-PRE)/(1-PRE);
    precision = TP/(TP+FP);
    recall = TP/(TP+FN);

end