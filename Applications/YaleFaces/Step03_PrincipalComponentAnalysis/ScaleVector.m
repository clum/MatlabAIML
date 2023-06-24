function [y] = ScaleVector(u,yMin,yMax)

%Rescale a vector so the range are within min/max.  This is useful for
%scaling to be used with imshow
assert(yMax>=yMin)

%Begin calculations
uMin = min(u);
uMax = max(u);

delta = yMax - yMin;