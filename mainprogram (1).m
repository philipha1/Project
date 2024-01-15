function mainprogram
%This is the main program for calculating the velocity
%with respect to time for a free-falling parachutist

mass = 0.005; % In this slot input the mass of the droplet
gravity = 9.81; % In this slot input the gravitational constant 
Dragc = 0.5; % In this slot input the air draft coefficient
Vi = 0; % In this slot input the initial velocity of the parachutist 
P = 1.290; % In this slot input the density of the air
Area = 0.002; % In this slot input the area of cross section

ts = 0; % input the initial time
tf = 12; % The final time
dt = 0.1; % The time step between each measure of velocity


v2 = Cal_of_vel (ts,tf,dt,mass,Dragc,gravity,P,Area,Vi)
%Calculate the velocity of the first 12 seconds with the step size of 2s
%with the second equation

t= ts:dt:tf;
plot(t,v2)


hold off
end
