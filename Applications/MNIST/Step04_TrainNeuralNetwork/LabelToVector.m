function [d] = LabelToVector(label)

d = zeros(10,1);
d(label+1) = 1;