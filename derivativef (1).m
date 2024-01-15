function dvdt = derivativef(gravity, P ,Dragc, Area, mass, Vi)
%Function name: derivativef(gravity ,P, Dragc, Area, mass, Vr)

%Output:
%  dvdt = the change of velocity with respect to time (acceleration)

dvdt= gravity -((P*Dragc*Area*Vi^2)/(2*mass));
end