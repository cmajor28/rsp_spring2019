function [ acc ] = get_accuracy( TP, FP, FN, TN )
    acc = (TP+TN)/(TP+TN+FP+FN);
end

