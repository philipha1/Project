function vel = Cal_of_vel (ts, tf, dt, mass, Dragc, gravity, P, Area, Vi)
% velocity_profile(dt, ti, tf, vi, mass , Dragc)
% we need to calculate the velocity of a free-falling object using Euler's method
% numerical method
% Working:
t = ts;
V(1) = Vi;
i=1;

while(1)
%Infinite loop
    dvdt = derivativef(gravity, P , Dragc, Area, mass, V(i));
    V(i+1)=V(i) + dvdt * dt;
    t=t+dt;
    i=i+1;
    if t > tf, break, end
    %If t > tf break out from the loop
end
 
vel = V(1:length(V)-1);
% Delete V(i+1)
end 