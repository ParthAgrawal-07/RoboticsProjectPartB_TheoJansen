clear; close all; clc;

if exist("fsolve", "file") == 2
    fprintf("Solver: MATLAB fsolve.\n");
else
    fprintf("Solver: built-in damped Newton.\n");
end

%% 1) Parameters
L = struct();
L.names = ["L1","L2","L3","L4","L5","L6","L7","L8","L9","L10","L11","L12"];
L.val = [12.8, 45.0, 36.0, 32.8, 48.5, 41.5, 60.5, 43.5, 42.0, 43.0, 26.5, 54.5].';

sim = struct();
sim.N                    = 361;
sim.omega                = 2*pi/2.0;
sim.phaseOffset          = 0.0;
sim.gaitFrameRotationDeg = -11.4;
sim.showPaperFigure4         = true;
sim.showNinePatternValidation = true;
sim.runAnimation              = true;
sim.animationFrameStep        = 3;
sim.showDiagnosticPlots       = false;
sim.NAdvanced                 = 91;

fprintf("Modified Jansen gait trainer simulation\n");
fprintf("L1=%.2f cm, L4=%.2f cm, L8=%.2f cm\n", L.val(1), L.val(4), L.val(8));

%% 2) Main simulation
main = simulateJansenCycle(L.val, sim.N, sim.omega, sim.phaseOffset, [], ...
    sim.gaitFrameRotationDeg*pi/180);

fprintf("x-span=%.2f cm, y-span=%.2f cm\n", main.span(1), main.span(2));
fprintf("Area=%.2f cm^2, Max residual=%.3e cm\n\n", main.area, max(main.resnorm));

%% 3) Reference curve
targetSpan = [50.02; 12.81];
ref        = makeReferenceGait(main.gaitCycle, targetSpan);
refAligned = alignCurveToSimulation(ref, main.PE);

%% 4) Plots
if sim.showPaperFigure4
    plotPaperStyleFigure4(main, refAligned, L);
end
if sim.showNinePatternValidation
    plotNinePatternValidation(L.val, sim);
end
if sim.runAnimation
    animateJansenMechanism(main, L, sim.animationFrameStep);
end
if sim.showDiagnosticPlots
    plotEndEffector(main, refAligned, targetSpan, L);
    plotGaitCurves(main, refAligned);
    plotVelocityCurves(main);
    plotConstraintResiduals(main);
end

%% 5) Extra required plots
hVals = [0.7 0.85 1.0 1.15 1.3];
plot_effect_varying_h(main, targetSpan, hVals);
plot_jansen_one_revolution(main);

PE_original = compute_PE_original_theo_jansen(sim.N);
plot_gait_comparison_original_vs_simulated(PE_original, main.PE);

effectOfVarying_m_onTrajectory(L.val, sim);
disp("Done.");

%% ======================== LOCAL FUNCTIONS ========================

function out = simulateJansenCycle(L, N, omega, phaseOffset, qStart, frameRotation)
    if nargin < 6, frameRotation = 0; end

    crank     = linspace(0, 2*pi, N);
    t         = crank / omega;
    gaitCycle = 100 * crank / (2*pi);   % FIXED typo [1]

    solver = makeLoopSolver();

    theta    = nan(12, N);
    qHist    = nan(10, N);
    exitflag = nan(1, N);
    resnorm  = nan(1, N);

    P0 = nan(2,N); P1=P0; P2=P0; P3=P0; P4=P0;
    P5 = P0; P6=P0; PE=P0;

    q = qStart;

    for k = 1:N
        theta1 = crank(k) + phaseOffset;

        if isempty(q) || any(~isfinite(q))
            q = geometricInitialGuess(L, theta1);
        end

        loopFun = @(qq) loopEquations(qq, theta1, L);
        loopJac = @(qq) loopJacobian(qq, L);

        [qCandidate, fval, flag] = solveLoopAngles(loopFun, loopJac, q, solver);

        if flag <= 0 || norm(fval) > 1e-6
            qGeom = geometricInitialGuess(L, theta1);
            [qCandidate2, fval2, flag2] = solveLoopAngles(loopFun, loopJac, qGeom, solver);
            if norm(fval2) < norm(fval)
                qCandidate = qCandidate2; fval = fval2; flag = flag2;
            end
        end

        q             = unwrapNear(qCandidate(:), q(:));
        qHist(:,k)    = q;
        exitflag(k)   = flag;
        resnorm(k)    = norm(fval);

        th      = nan(12,1);
        th(1)   = theta1;
        th(2)   = q(1);  th(3)  = q(2);  th(4)  = 0;
        th(5)   = q(3);  th(6)  = q(4);  th(7)  = q(5);
        th(8)   = q(6);  th(9)  = q(7);  th(10) = q(8);
        th(11)  = q(9);  th(12) = q(10);
        theta(:,k) = th;

        pos         = positionsFromAngles(L, th);
        P0(:,k)=pos.P0; P1(:,k)=pos.P1; P2(:,k)=pos.P2;
        P3(:,k)=pos.P3; P4(:,k)=pos.P4; P5(:,k)=pos.P5;
        P6(:,k)=pos.P6; PE(:,k)=pos.PE;
    end

    raw = struct("P0",P0,"P1",P1,"P2",P2,"P3",P3, ...
                 "P4",P4,"P5",P5,"P6",P6,"PE",PE);

    if abs(frameRotation) > 0
        R  = [cos(frameRotation), -sin(frameRotation); ...
              sin(frameRotation),  cos(frameRotation)];
        P0=R*P0; P1=R*P1; P2=R*P2; P3=R*P3; P4=R*P4;
        P5=R*P5; P6=R*P6; PE=R*PE;
    end

    vx    = gradient(PE(1,:), t);
    vy    = gradient(PE(2,:), t);
    speed = hypot(vx, vy);

    out = struct();
    out.t=t; out.crank=crank; out.gaitCycle=gaitCycle;
    out.theta=theta; out.qHist=qHist;
    out.exitflag=exitflag; out.resnorm=resnorm;
    out.P0=P0; out.P1=P1; out.P2=P2; out.P3=P3; out.P4=P4;
    out.P5=P5; out.P6=P6; out.PE=PE;
    out.rawMechanismFrame=raw;
    out.frameRotation_rad=frameRotation;
    out.vx=vx; out.vy=vy; out.speed=speed;
    out.span = [spanOf(PE(1,:)); spanOf(PE(2,:))];
    out.area = abs(polyarea(PE(1,:), PE(2,:)));
end

function F = loopEquations(q, theta1, L)
    u = @(a) [cos(a); sin(a)];
    theta2=q(1); theta3=q(2); theta5=q(3); theta6=q(4);
    theta7=q(5); theta8=q(6); theta9=q(7); theta10=q(8);
    theta11=q(9); theta12=q(10);

    e1=u(theta1); e2=u(theta2); e3=u(theta3); e4=[1;0];
    e5=u(theta5); e6=u(theta6); e7=u(theta7); e8=u(theta8);
    e9=u(theta9); e10=u(theta10); e11=u(theta11); e12=u(theta12);

    F = [L(1)*e1+L(2)*e2-L(3)*e3-L(4)*e4;
         L(1)*e1+L(7)*e7-L(8)*e8-L(4)*e4;
         L(3)*e3+L(5)*e5-L(6)*e6;
         L(8)*e8+L(9)*e9-L(6)*e6-L(10)*e10;
         L(11)*e11+L(12)*e12-L(9)*e9];
end

function J = loopJacobian(q, L)
    d = @(a) [-sin(a); cos(a)];
    theta2=q(1); theta3=q(2); theta5=q(3); theta6=q(4);
    theta7=q(5); theta8=q(6); theta9=q(7); theta10=q(8);
    theta11=q(9); theta12=q(10);

    J = zeros(10,10);
    J(1:2,1)=L(2)*d(theta2);   J(1:2,2)=-L(3)*d(theta3);
    J(3:4,5)=L(7)*d(theta7);   J(3:4,6)=-L(8)*d(theta8);
    J(5:6,2)=L(3)*d(theta3);   J(5:6,3)=L(5)*d(theta5);
    J(5:6,4)=-L(6)*d(theta6);
    J(7:8,4)=-L(6)*d(theta6);  J(7:8,6)=L(8)*d(theta8);
    J(7:8,7)=L(9)*d(theta9);   J(7:8,8)=-L(10)*d(theta10);
    J(9:10,7)=-L(9)*d(theta9); J(9:10,9)=L(11)*d(theta11);
    J(9:10,10)=L(12)*d(theta12);
end

function solver = makeLoopSolver()
    solver.useFsolve    = exist("fsolve","file")==2;
    solver.tolF         = 1e-10;
    solver.tolStep      = 1e-11;
    solver.maxIter      = 80;
    solver.maxLineSearch= 16;
    if solver.useFsolve
        solver.options = optimoptions("fsolve","Display","off", ...
            "FunctionTolerance",1e-10,"StepTolerance",1e-11, ...
            "OptimalityTolerance",1e-10,"MaxIterations",120, ...
            "MaxFunctionEvaluations",1500);
    else
        solver.options = [];
    end
end

function [x, F, exitflag] = solveLoopAngles(fun, jac, x0, solver)
    if solver.useFsolve
        [x,F,exitflag] = fsolve(fun, x0, solver.options);
        return;
    end
    [x,F,exitflag] = dampedNewtonSolve(fun, jac, x0, solver);
end

function [x, F, exitflag] = dampedNewtonSolve(fun, jac, x0, solver)
    x=x0(:); F=fun(x); nrm=norm(F); exitflag=0;
    for iter = 1:solver.maxIter %#ok<NASGU>
        if nrm < solver.tolF, exitflag=1; return; end
        J    = jac(x);
        step = -J\F;
        if any(~isfinite(step)) || norm(step)>5, step=-pinv(J)*F; end
        alpha=1.0; accepted=false;
        for ls = 1:solver.maxLineSearch %#ok<NASGU>
            xt=x+alpha*step; Ft=fun(xt); nt=norm(Ft);
            if nt<nrm || nt<solver.tolF
                x=xt; F=Ft; nrm=nt; accepted=true; break;
            end
            alpha=0.5*alpha;
        end
        if ~accepted, x=x+alpha*step; F=fun(x); nrm=norm(F); end
        if norm(alpha*step)<solver.tolStep*(1+norm(x))
            exitflag=double(nrm<1e-7); return;
        end
    end
end

function pos = positionsFromAngles(L, th)
    u  = @(a) [cos(a); sin(a)];
    P0 = [0;0];
    P3 = [L(4);0];
    P1 = P0+L(1)*u(th(1));
    P2 = P1+L(2)*u(th(2));
    P5 = P1+L(7)*u(th(7));
    P4 = P2+L(5)*u(th(5));
    P6 = P5+L(9)*u(th(9));
    PE = P5+L(11)*u(th(11));
    pos = struct("P0",P0,"P1",P1,"P2",P2,"P3",P3, ...
                 "P4",P4,"P5",P5,"P6",P6,"PE",PE);
end

function q = geometricInitialGuess(L, theta1)
    P0=[0;0]; P3=[L(4);0];
    P1=P0+L(1)*[cos(theta1);sin(theta1)];
    P2c=circleIntersections(P1,L(2),P3,L(3)); P2=pickBy(P2c,"maxY");
    P5c=circleIntersections(P1,L(7),P3,L(8)); P5=pickBy(P5c,"minY");
    P4c=circleIntersections(P2,L(5),P3,L(6)); P4=pickBy(P4c,"maxX");
    P6c=circleIntersections(P5,L(9),P4,L(10)); P6=pickBy(P6c,"maxX");
    if P6(2)>min(P4(2),P5(2)), P6=pickBy(P6c,"minY"); end
    PEc=circleIntersections(P5,L(11),P6,L(12)); PE=pickBy(PEc,"minY");
    ang=@(a,b) atan2(b(2)-a(2),b(1)-a(1));
    q=[ang(P1,P2);ang(P3,P2);ang(P2,P4);ang(P3,P4); ...
       ang(P1,P5);ang(P3,P5);ang(P5,P6);ang(P4,P6); ...
       ang(P5,PE);ang(PE,P6)];
end

function pts = circleIntersections(c1, r1, c2, r2)
    dvec=c2-c1; d=norm(dvec);
    if d<eps, error("Coincident centers."); end
    if d>r1+r2 || d<abs(r1-r2), error("Non-assemblable."); end
    a=(r1^2-r2^2+d^2)/(2*d);
    h=sqrt(max(r1^2-a^2,0));
    ex=dvec/d; ey=[-ex(2);ex(1)];
    p=c1+a*ex;
    pts=[p+h*ey, p-h*ey];
end

function p = pickBy(pts, mode)
    switch mode
        case "maxY", [~,idx]=max(pts(2,:));
        case "minY", [~,idx]=min(pts(2,:));
        case "maxX", [~,idx]=max(pts(1,:));
        case "minX", [~,idx]=min(pts(1,:));
        otherwise,   error("Unknown mode.");
    end
    p=pts(:,idx);
end

function q = unwrapNear(qNew, qOld)
    if isempty(qOld)||any(~isfinite(qOld)), q=qNew; return; end
    q=qNew+2*pi*round((qOld-qNew)/(2*pi));
end

function s = spanOf(v)
    s=max(v(:))-min(v(:));
end

function ref = makeReferenceGait(gaitCycle, span)
    s=gaitCycle(:).'/100;
    knot  =[0.00 0.08 0.16 0.28 0.42 0.58 0.72 0.86 1.00];
    xShape=[0.48 0.42 0.25 -0.15 -0.50 -0.35 -0.10 0.20 0.48];
    yShape=[0.55 0.95 1.00  0.35  0.15  0.06  0.00 0.18 0.55];
    x=pchip(knot,xShape,s); y=pchip(knot,yShape,s);
    x=span(1)*(x-min(x))/spanOf(x); y=span(2)*(y-min(y))/spanOf(y);
    x=x-mean(x); y=y-min(y);
    ref=[x;y];
end

function refAligned = alignCurveToSimulation(ref, PE)
    refAligned=ref;
    refAligned(1,:)=ref(1,:)-mean(ref(1,:))+mean(PE(1,:));
    refAligned(2,:)=ref(2,:)-min(ref(2,:))+min(PE(2,:));
end

%% -------------------- PLOT FUNCTIONS --------------------

function plotPaperStyleFigure4(main, ref, L)
    fig=figure("Name","Fig. 4 - Modified Jansen","Color","w","Position",[80 80 1120 560]);

    axes(fig,"Position",[0.06 0.16 0.53 0.76]);
    k=22;
    drawPaperMechanism(main, k);
    hold on;
    idx=1:5:numel(main.gaitCycle);

    % CHANGED: reference markers = green diamonds, PE = blue squares
    plot(ref(1,idx),      ref(2,idx),      "gd","MarkerSize",5.5,"LineWidth",1.2);
    plot(main.PE(1,idx),  main.PE(2,idx),  "bs","MarkerSize",4.5,"LineWidth",1.2);

    lengthText=makeLengthListText(L.val);
    xT=min([main.P0(1,:),main.PE(1,:)])-6;
    yT=max([main.P2(2,:),main.P4(2,:)])-4;
    text(xT,yT,lengthText,"FontSize",9,"FontName","Consolas", ...
        "VerticalAlignment","top","Interpreter","none");
    axis equal; axis off;
    title("Parameterized 12-link modified Jansen mechanism","FontSize",11,"FontWeight","normal");

    axes(fig,"Position",[0.67 0.58 0.28 0.30]);
    predX=main.PE(1,:)-mean(main.PE(1,:));
    refX=ref(1,:)-mean(ref(1,:));
    % CHANGED: reference = green dashed, predicted = blue solid
    plot(main.gaitCycle,refX, "g--","LineWidth",1.5); hold on;
    plot(main.gaitCycle,predX,"b-", "LineWidth",1.8);
    grid on; xlim([0 100]); ylim([-40 40]);
    ylabel("x-axis (cm)"); set(gca,"FontSize",9);

    axes(fig,"Position",[0.67 0.20 0.28 0.30]);
    predY=main.PE(2,:)-min(main.PE(2,:));
    refY=ref(2,:)-min(ref(2,:));
    plot(main.gaitCycle,refY, "g--","LineWidth",1.5); hold on;
    plot(main.gaitCycle,predY,"b-", "LineWidth",1.8);
    grid on; xlim([0 100]); ylim([0 15]);
    xlabel("Gait Cycle (%)"); ylabel("y-axis (cm)");
    legend("meta-trajectory","predicted trajectory","Location","northeast","FontSize",8);
    set(gca,"FontSize",9);

    caption=sprintf("Fig. 4  x-span=%.2f cm, y-span=%.2f cm.", main.span(1), main.span(2));
    annotation(fig,"textbox",[0.08 0.015 0.86 0.10],"String",caption, ...
        "EdgeColor","none","FontSize",10,"FontWeight","bold");
end

function drawPaperMechanism(main, k)
    linePairs={
        "P0","P1","L_1"; "P1","P2","L_2"; "P3","P2","L_3"; "P0","P3","L_4";
        "P2","P4","L_5"; "P3","P4","L_6"; "P1","P5","L_7"; "P3","P5","L_8";
        "P5","P6","L_9"; "P4","P6","L_{10}"; "P5","PE","L_{11}"; "PE","P6","L_{12}"};

    % CHANGED link color: dark teal [0 0.45 0.45]
    linkColor=[0 0.45 0.45];
    cla; hold on;
    for i=1:size(linePairs,1)
        A=main.(linePairs{i,1})(:,k);
        B=main.(linePairs{i,2})(:,k);
        plot([A(1),B(1)],[A(2),B(2)],"-","Color",linkColor,"LineWidth",2.0);
        mid=0.52*A+0.48*B;
        text(mid(1),mid(2),linePairs{i,3},"FontSize",10,"FontWeight","bold", ...
            "Interpreter","tex","Color",[0.1 0.1 0.5]);
    end

    pointNames=["P0","P1","P2","P3","P4","P5","P6","PE"];
    P=zeros(2,numel(pointNames));
    for i=1:numel(pointNames)
        P(:,i)=main.(pointNames(i))(:,k);
    end
    % CHANGED joint color: filled orange circles
    plot(P(1,:),P(2,:),"o","Color",[0.85 0.33 0.10], ...
        "MarkerFaceColor",[1.0 0.6 0.2],"MarkerSize",7);

    xM=8; yM=8;
    xlim([min(P(1,:))-xM, max(P(1,:))+xM]);
    ylim([min(P(2,:))-yM, max(P(2,:))+yM]);
end

function txt = makeLengthListText(L)
    lines=strings(14,1); lines(1)="Unit: cm";
    for i=1:12, lines(i+1)=sprintf("L%-2d = %4.1f",i,L(i)); end
    txt=strjoin(lines,newline);
end

function validation = plotNinePatternValidation(Lnom, sim)
    xSpanGrid=[56 53 50; 53 50 47; 50 47 44];
    ySpanGrid=[14.2 13.6 13.0; 13.6 13.0 12.4; 13.0 12.4 11.8];
    L1Grid=Lnom(1)+[0.45 0.25 0.05; 0.25 0.00 -0.20; 0.05 -0.20 -0.45];
    L4Grid=Lnom(4)+[-0.90 -0.50 -0.10; -0.50 0.00 0.50; -0.10 0.50 0.90];
    L8Grid=Lnom(8)+[0.90 0.50 0.10; 0.50 0.00 -0.50; 0.10 -0.50 -0.90];

    figure("Name","3x3 validation grid","Color","w","Position",[120 90 1040 620]);
    tl=tiledlayout(3,3,"Padding","compact","TileSpacing","compact");

    validation=struct();
    validation.patterns=repmat(struct("L",[],"reference",[],"simulation",[], ...
        "rmse",NaN,"span",[],"ok",false,"msg",""),3,3);

    for row=1:3
        for col=1:3
            ax=nexttile; hold(ax,"on"); grid(ax,"on");
            xlim(ax,[0 70]); ylim(ax,[0 20]); set(ax,"FontSize",8);
            if row<3, set(ax,"XTickLabel",[]); else, xlabel(ax,"x (cm)"); end
            if col>1, set(ax,"YTickLabel",[]); else, ylabel(ax,"y (cm)"); end

            Lc=Lnom; Lc(1)=L1Grid(row,col); Lc(4)=L4Grid(row,col); Lc(8)=L8Grid(row,col);

            try
                out=simulateJansenCycle(Lc,sim.NAdvanced,sim.omega,sim.phaseOffset,[], ...
                    sim.gaitFrameRotationDeg*pi/180);
                rf=makeReferenceGait(out.gaitCycle,[xSpanGrid(row,col);ySpanGrid(row,col)]);
                rf=alignCurveToSimulation(rf,out.PE);
                [simP,refP]=panelNormalizeCurves(out.PE,rf);
                rmse=sqrt(mean(sum((simP-refP).^2,1)));
                mi=1:2:numel(out.gaitCycle);
                % CHANGED: reference=purple x, simulation=orange circle
                plot(ax,refP(1,mi),refP(2,mi),"x","Color",[0.5 0 0.8],"MarkerSize",5,"LineWidth",1.0);
                plot(ax,simP(1,mi),simP(2,mi),"o","Color",[0.9 0.4 0.0],"MarkerSize",4,"LineWidth",1.0);
                text(ax,5,17,sprintf("RMSE=%.2f",rmse),"FontSize",11);
                validation.patterns(row,col).L=Lc;
                validation.patterns(row,col).reference=refP;
                validation.patterns(row,col).simulation=simP;
                validation.patterns(row,col).rmse=rmse;
                validation.patterns(row,col).span=out.span;
                validation.patterns(row,col).ok=true;
            catch ME
                text(ax,3,12,"FAILED","Color","r","FontWeight","bold");
                text(ax,3,9,sprintf("row=%d col=%d",row,col),"Color","r");
                validation.patterns(row,col).ok=false;
                validation.patterns(row,col).msg=ME.message;
            end
            if row==1&&col==1
                legend(ax,"reference","simulation","Location","northwest","FontSize",7);
            end
        end
    end
    title(tl,"Reference-vs-simulation for nine gait envelopes","FontSize",12,"FontWeight","bold");
end

function [simPanel, refPanel] = panelNormalizeCurves(simCurve, refCurve)
    simPanel=simCurve; refPanel=refCurve;
    xmin=min([simPanel(1,:),refPanel(1,:)]);
    ymin=min([simPanel(2,:),refPanel(2,:)]);
    simPanel(1,:)=simPanel(1,:)-xmin+8; refPanel(1,:)=refPanel(1,:)-xmin+8;
    simPanel(2,:)=simPanel(2,:)-ymin+1.5; refPanel(2,:)=refPanel(2,:)-ymin+1.5;
end

function animateJansenMechanism(main, L, frameStep)
    if nargin<3, frameStep=3; end
    if ~usejava("desktop"), disp("Animation skipped."); return; end
    fig=figure("Name","Animation","Color","w","Position",[160 120 900 560]);
    ax=axes("Parent",fig);
    allX=[main.P0(1,:),main.P1(1,:),main.P2(1,:),main.P3(1,:), ...
          main.P4(1,:),main.P5(1,:),main.P6(1,:),main.PE(1,:)];
    allY=[main.P0(2,:),main.P1(2,:),main.P2(2,:),main.P3(2,:), ...
          main.P4(2,:),main.P5(2,:),main.P6(2,:),main.PE(2,:)];
    xLim=[min(allX)-8,max(allX)+8]; yLim=[min(allY)-8,max(allY)+8];

    for k=1:frameStep:numel(main.gaitCycle)
        if ~isvalid(fig), break; end
        cla(ax); axes(ax); %#ok<LAXES>
        % CHANGED animation link color: deep blue [0.05 0.20 0.70]
        drawMechanism(main, k, [0.05 0.20 0.70]); hold on;
        % CHANGED trail color: magenta -> cyan
        plot(main.PE(1,1:k),main.PE(2,1:k),"-","Color",[0 0.75 0.75],"LineWidth",2.2);
        % CHANGED current foot marker: orange
        plot(main.PE(1,k),main.PE(2,k),"o","Color",[0.9 0.4 0],"MarkerFaceColor",[1 0.6 0],"MarkerSize",8);
        grid on; axis equal; xlim(xLim); ylim(yLim);
        xlabel("x (cm)"); ylabel("y (cm)");
        title(sprintf("Crank = %.1f deg", main.crank(k)*180/pi));
        text(xLim(1)+2,yLim(2)-4, ...
            sprintf("L1=%.2f, L4=%.2f, L8=%.2f cm",L.val(1),L.val(4),L.val(8)), ...
            "BackgroundColor","w","Margin",4);
        drawnow; pause(0.005);
    end
end

function drawMechanism(main, k, color)
    linePairs={"P0","P1";"P1","P2";"P3","P2";"P0","P3"; ...
               "P1","P5";"P3","P5";"P2","P4";"P3","P4"; ...
               "P5","P6";"P4","P6";"P5","PE";"PE","P6"};
    hold on;
    for i=1:size(linePairs,1)
        A=main.(linePairs{i,1})(:,k);
        B=main.(linePairs{i,2})(:,k);
        plot([A(1),B(1)],[A(2),B(2)],"-","Color",color,"LineWidth",1.4);
    end
    pts=[main.P0(:,k),main.P1(:,k),main.P2(:,k),main.P3(:,k), ...
         main.P4(:,k),main.P5(:,k),main.P6(:,k),main.PE(:,k)];
    % CHANGED joint color: yellow fill with dark border
    plot(pts(1,:),pts(2,:),"o","Color",[0.2 0.2 0.2], ...
        "MarkerFaceColor",[1 0.85 0],"MarkerSize",5);
end

function plotEndEffector(main, ref, targetSpan, L)
    figure("Name","End-effector trajectory","Color","w");
    % CHANGED: trajectory = navy blue, reference = orange dashed
    plot(main.PE(1,:),main.PE(2,:),"-","Color",[0 0.2 0.7],"LineWidth",2.4); hold on;
    plot(ref(1,:),ref(2,:),"--","Color",[0.9 0.4 0],"LineWidth",1.7);
    plot(main.PE(1,1),main.PE(2,1),"p","Color",[0 0.5 0], ...
        "MarkerFaceColor",[0 0.8 0],"MarkerSize",9);
    grid on; axis equal;
    xlabel("x (cm)"); ylabel("y (cm)");
    title("Modified Jansen end-effector trajectory");
    legend("simulated",sprintf("reference %.2fx%.2f cm",targetSpan(1),targetSpan(2)), ...
        "start","Location","best");
    text(min(main.PE(1,:)),max(main.PE(2,:)), ...
        sprintf("L1=%.2f, L4=%.2f, L8=%.2f cm",L.val(1),L.val(4),L.val(8)), ...
        "VerticalAlignment","top","BackgroundColor","w","Margin",4);
end

function plotGaitCurves(main, ref)
    figure("Name","Gait-cycle curves","Color","w");
    tiledlayout(2,1,"Padding","compact","TileSpacing","compact");

    nexttile;
    % CHANGED: simulated=navy, reference=orange dashed
    plot(main.gaitCycle,main.PE(1,:),"-","Color",[0 0.2 0.7],"LineWidth",2.2); hold on;
    plot(main.gaitCycle,ref(1,:),"--","Color",[0.9 0.4 0],"LineWidth",1.5);
    grid on; ylabel("x (cm)"); title("Horizontal motion over gait cycle");
    legend("simulated","reference","Location","best");

    nexttile;
    plot(main.gaitCycle,main.PE(2,:),"-","Color",[0 0.2 0.7],"LineWidth",2.2); hold on;
    plot(main.gaitCycle,ref(2,:),"--","Color",[0.9 0.4 0],"LineWidth",1.5);
    grid on; xlabel("gait cycle (%)"); ylabel("y (cm)");
    title("Vertical motion over gait cycle");
end

function plotVelocityCurves(main)
    figure("Name","Velocity curves","Color","w");
    tiledlayout(3,1,"Padding","compact","TileSpacing","compact");
    % CHANGED: teal color for velocity plots
    c=[0 0.55 0.55];
    nexttile; plot(main.gaitCycle,main.vx,"-","Color",c,"LineWidth",1.8);
    grid on; ylabel("vx (cm/s)"); title("End-effector velocity");
    nexttile; plot(main.gaitCycle,main.vy,"-","Color",c,"LineWidth",1.8);
    grid on; ylabel("vy (cm/s)");
    nexttile; plot(main.gaitCycle,main.speed,"-","Color",[0.7 0 0.35],"LineWidth",1.8);
    grid on; xlabel("gait cycle (%)"); ylabel("|v| (cm/s)");
end

function plotConstraintResiduals(main)
    figure("Name","Loop residuals","Color","w");
    semilogy(main.gaitCycle,main.resnorm+eps,"-","Color",[0.4 0 0.6],"LineWidth",1.8);
    grid on; xlabel("gait cycle (%)"); ylabel("||residual||_2 (cm)");
    title("Vector-loop closure residual");
end

%% -------------------- EXTRA PLOTS --------------------

function plot_effect_varying_h(main, targetSpan, hVals)
    ref0=makeReferenceGait(main.gaitCycle,targetSpan);
    ref0=alignCurveToSimulation(ref0,main.PE);

    figure("Color","w","Name","Effect of varying h");
    tiledlayout(1,2,"Padding","compact","TileSpacing","compact");

    nexttile; hold on; grid on; axis equal;
    xlabel("x (cm)"); ylabel("y (cm)");
    title("Reference gait envelopes (varying h)");
    cmap=turbo(numel(hVals));
    for i=1:numel(hVals)
        sp=targetSpan; sp(2)=hVals(i)*targetSpan(2);
        rf=makeReferenceGait(main.gaitCycle,sp);
        rf=alignCurveToSimulation(rf,main.PE);
        plot(rf(1,:),rf(2,:),"LineWidth",2.0,"Color",cmap(i,:));
    end
    legend(compose("h=%.2f",hVals),"Location","best");

    nexttile; hold on; grid on; axis equal;
    xlabel("x (cm)"); ylabel("y (cm)");
    title("Simulated PE with baseline reference");
    % CHANGED: PE = navy, reference = orange
    plot(main.PE(1,:),main.PE(2,:),"-","Color",[0 0.2 0.7],"LineWidth",2.4);
    plot(ref0(1,:),ref0(2,:),"--","Color",[0.9 0.4 0],"LineWidth",1.8);
    legend("Simulated PE","Baseline reference","Location","best");
end

function plot_jansen_one_revolution(main)
    figure("Color","w","Name","Foot trajectory (one revolution)");
    % CHANGED: trajectory = dark green
    plot(main.PE(1,:),main.PE(2,:),"-","Color",[0 0.5 0.15],"LineWidth",2.4); hold on;
    plot(main.PE(1,1),main.PE(2,1),"p","Color",[0.8 0 0], ...
        "MarkerFaceColor",[1 0.2 0.2],"MarkerSize",10);
    grid on; axis equal;
    xlabel("x (cm)"); ylabel("y (cm)");
    title("Foot-point trajectory over one full crank revolution [2]");
    legend("PE trajectory","start","Location","best");
end

function plot_gait_comparison_original_vs_simulated(PE_original, mainPE)
    orig=PE_original; sim=mainPE;
    orig(1,:)=orig(1,:)-mean(orig(1,:)); sim(1,:)=sim(1,:)-mean(sim(1,:));
    orig(2,:)=orig(2,:)-min(orig(2,:));  sim(2,:)=sim(2,:)-min(sim(2,:));
    simR=resampleCurve(sim,size(orig,2));
    rmse=sqrt(mean(sum((simR-orig).^2,1)));

    figure("Color","w","Name","Gait comparison: original vs simulated");
    % CHANGED: original = black, simulated = red dashed
    plot(orig(1,:),orig(2,:),"k-","LineWidth",2.4); hold on;
    plot(simR(1,:),simR(2,:),"--","Color",[0.85 0.1 0.1],"LineWidth",2.4);
    grid on; axis equal;
    xlabel("x (cm)"); ylabel("y (cm)");
    title(sprintf("Trajectory comparison (RMSE=%.2f cm)",rmse));
    legend("Original Theo Jansen","Simulated modified-Jansen","Location","best");
end

function Y = resampleCurve(X, N)
    M=size(X,2); t0=linspace(0,1,M); t1=linspace(0,1,N);
    Y=[interp1(t0,X(1,:),t1,"pchip"); interp1(t0,X(2,:),t1,"pchip")];
end

function effectOfVarying_m_onTrajectory(Lnom, sim)
    mVals=linspace(0.96,1.04,9);
    idxScale=[1 4 8];

    figure("Color","w","Name","Effect of varying m");
    tiledlayout(1,2,"Padding","compact","TileSpacing","compact");
    nexttile; hold on; grid on; axis equal;
    xlabel("x (cm)"); ylabel("y (cm)"); title("PE trajectory for different m");
    nexttile; hold on; grid on;
    xlabel("m"); ylabel("span (cm)"); title("Span vs m");

    spanX=nan(size(mVals)); spanY=nan(size(mVals));
    cmap=turbo(numel(mVals));

    for i=1:numel(mVals)
        m=mVals(i); Lc=Lnom; Lc(idxScale)=m*Lnom(idxScale);
        out=simulateJansenCycle(Lc,sim.NAdvanced,sim.omega,sim.phaseOffset,[], ...
            sim.gaitFrameRotationDeg*pi/180);
        nexttile(1);
        plot(out.PE(1,:),out.PE(2,:),"Color",cmap(i,:),"LineWidth",1.6);
        text(out.PE(1,1),out.PE(2,1),sprintf("m=%.3f",m),"Color",cmap(i,:),"FontSize",7);
        spanX(i)=out.span(1); spanY(i)=out.span(2);
    end

    nexttile(2);
    % CHANGED: x-span = navy circles, y-span = orange squares
    plot(mVals,spanX,"o-","Color",[0 0.2 0.7],"LineWidth",1.8,"DisplayName","x-span");
    plot(mVals,spanY,"s-","Color",[0.9 0.4 0],"LineWidth",1.8,"DisplayName","y-span");
    legend("Location","best");
end

function plotNineAngle3x3Grid(main)
% Generate a 3x3 grid of subplots showing the full mechanism pose
% at 9 equal crank angles (0 to 360 deg).
% Each subplot shows: all 12 links, all joints, foot-point PE marked,
% and the complete PE trajectory as background reference.
%
% The end-effector path PE is determined by crank angle and link config [3].
% Recording over a complete input cycle is the core Part-B task [2].

    N_angles = 9;

    % Pick 9 equally spaced indices across one full revolution
    allIdx   = round(linspace(1, numel(main.gaitCycle), N_angles + 1));
    allIdx(end) = [];   % remove duplicate of 360 = 0

    % Color map: each pose gets a distinct color
    cmap = hsv(N_angles);

    figure("Color","w", ...
           "Name","Modified Jansen: 9 equal crank angles (3x3 grid)", ...
           "Position",[60 60 1200 900]);

    tl = tiledlayout(3, 3, "Padding","compact", "TileSpacing","compact");
    title(tl, ...
        "Modified Jansen mechanism pose at 9 equal crank angles (0° – 360°)", ...
        "FontSize", 13, "FontWeight","bold");

    % Compute fixed axis limits from all positions so all subplots share same scale
    allX = [main.P0(1,:), main.P1(1,:), main.P2(1,:), main.P3(1,:), ...
            main.P4(1,:), main.P5(1,:), main.P6(1,:), main.PE(1,:)];
    allY = [main.P0(2,:), main.P1(2,:), main.P2(2,:), main.P3(2,:), ...
            main.P4(2,:), main.P5(2,:), main.P6(2,:), main.PE(2,:)];
    xLim = [min(allX)-6,  max(allX)+6 ];
    yLim = [min(allY)-6,  max(allY)+6 ];

    for i = 1:N_angles
        k   = allIdx(i);
        deg = main.crank(k) * 180/pi;
        col = cmap(i,:);

        ax = nexttile;
        hold(ax,"on");
        grid(ax,"on");
        axis(ax,"equal");
        xlim(ax, xLim);
        ylim(ax, yLim);
        set(ax,"FontSize",8);
        xlabel(ax,"x (cm)");
        ylabel(ax,"y (cm)");
        title(ax, sprintf("\\theta_1 = %.1f°  |  GC = %.1f%%", ...
            deg, main.gaitCycle(k)), "FontSize",9);

        %% (a) Full PE trajectory as faint grey background
        plot(ax, main.PE(1,:), main.PE(2,:), ...
            "-","Color",[0.75 0.75 0.75],"LineWidth",1.0, ...
            "HandleVisibility","off");

        %% (b) Draw all 12 links at this crank angle
        drawLinks(ax, main, k, col);

        %% (c) Draw all joints
        drawJoints(ax, main, k, col);

        %% (d) Highlight foot point PE with large marker
        plot(ax, main.PE(1,k), main.PE(2,k), ...
            "p","Color",col, ...
            "MarkerFaceColor",col, ...
            "MarkerSize",12, ...
            "DisplayName",sprintf("PE @ %.0f°", deg));

        %% (e) Mark ground pivots P0 and P3 with triangles
        plot(ax, main.P0(1,k), main.P0(2,k), ...
            "^","Color",[0 0 0],"MarkerFaceColor",[0.3 0.3 0.3],"MarkerSize",7, ...
            "HandleVisibility","off");
        plot(ax, main.P3(1,k), main.P3(2,k), ...
            "^","Color",[0 0 0],"MarkerFaceColor",[0.3 0.3 0.3],"MarkerSize",7, ...
            "HandleVisibility","off");

        %% (f) Label key joints
        labelJoint(ax, main.P0(:,k), "P0",  col);
        labelJoint(ax, main.P1(:,k), "P1",  col);
        labelJoint(ax, main.PE(:,k), "PE",  col);

        %% (g) Crank arm highlight (P0 -> P1) in thick color
        plot(ax, [main.P0(1,k), main.P1(1,k)], ...
                 [main.P0(2,k), main.P1(2,k)], ...
            "-","Color",col,"LineWidth",3.0, ...
            "HandleVisibility","off");

        %% (h) Show foot position coordinates in corner
        xPE = main.PE(1,k);
        yPE = main.PE(2,k);
        text(ax, xLim(1)+1.5, yLim(2)-3.0, ...
            sprintf("PE=(%.1f,%.1f)", xPE, yPE), ...
            "FontSize",7,"Color",col, ...
            "BackgroundColor","w","Margin",2);

        legend(ax, sprintf("PE @ %.0f°", deg), ...
            "Location","southwest","FontSize",7);
    end
end

%% ---- Helper: draw all 12 links at frame k ----
function drawLinks(ax, main, k, col)
    linePairs = {
        "P0","P1";  "P1","P2";  "P3","P2";  "P0","P3";
        "P1","P5";  "P3","P5";  "P2","P4";  "P3","P4";
        "P5","P6";  "P4","P6";  "P5","PE";  "PE","P6"};

    for i = 1:size(linePairs,1)
        A = main.(linePairs{i,1})(:,k);
        B = main.(linePairs{i,2})(:,k);

        % Ground link P0-P3: draw as thick grey base
        if strcmp(linePairs{i,1},"P0") && strcmp(linePairs{i,2},"P3")
            plot(ax,[A(1),B(1)],[A(2),B(2)], ...
                "-","Color",[0.4 0.4 0.4],"LineWidth",2.5, ...
                "HandleVisibility","off");
        else
            plot(ax,[A(1),B(1)],[A(2),B(2)], ...
                "-","Color",col,"LineWidth",1.8, ...
                "HandleVisibility","off");
        end
    end
end

%% ---- Helper: draw all joints at frame k ----
function drawJoints(ax, main, k, col)
    pointNames = ["P0","P1","P2","P3","P4","P5","P6","PE"];
    pts = zeros(2, numel(pointNames));
    for i = 1:numel(pointNames)
        pts(:,i) = main.(pointNames(i))(:,k);
    end

    % Regular joints: white fill with colored border
    plot(ax, pts(1,:), pts(2,:), ...
        "o","Color",col, ...
        "MarkerFaceColor","w", ...
        "MarkerSize",5, ...
        "HandleVisibility","off");
end

%% ---- Helper: label one joint ----
function labelJoint(ax, pt, name, col)
    text(ax, pt(1)+0.6, pt(2)+0.6, name, ...
        "FontSize", 7, ...
        "Color",    col, ...
        "FontWeight","bold", ...
        "HandleVisibility","off");
end


plotNineAngle3x3Grid(main);

%% -------------------- ORIGINAL THEO JANSEN (classic) --------------------

%% -------------------- ORIGINAL THEO JANSEN (classic) --------------------

function PE_original = compute_PE_original_theo_jansen(N)
% Computes classic Theo Jansen leg foot-point trajectory over one crank revolution.
% Uses the well-known 11-constant set (Jansen 2011), scaled to cm.
% All proportions are chosen so the mechanism assembles for all crank angles.

    if nargin < 1, N = 361; end
    theta = linspace(0, 2*pi, N);
    p     = jansenDefaultParams_cm();

    PE_original = nan(2, N);
    for k = 1:N
        try
            PE_original(:,k) = jansenFootPoint(theta(k), p);
        catch
            % leave NaN for non-assemblable angles (interpolated below)
        end
    end

    % Fill any NaN gaps by linear interpolation so downstream plots work
    for row = 1:2
        bad  = ~isfinite(PE_original(row,:));
        good = ~bad;
        if any(bad) && sum(good) > 3
            PE_original(row, bad) = interp1( ...
                find(good), PE_original(row, good), find(bad), "linear","extrap");
        end
    end
end

function p = jansenDefaultParams_cm()
% Classic Theo Jansen proportions from the well-known "holy numbers" (Jansen 2011).
% Original values in mm, converted to cm here (divide by 10).
% These are the proportions that are known to assemble for all crank angles.
%
%   a  = crank
%   b,c = upper four-bar coupler sides
%   d,e = lower diagonal
%   f,g = lower four-bar coupler sides
%   j,k = foot link sides
%
% Fixed pivots: O (crank centre), A (second ground pivot).
% Distance O-A = 38 mm = 3.8 cm.

    mm2cm = 0.1;

    p.a = 38.0 * mm2cm;   % crank length
    p.b = 41.5 * mm2cm;
    p.c = 39.3 * mm2cm;
    p.d = 40.1 * mm2cm;
    p.e = 55.8 * mm2cm;
    p.f = 39.4 * mm2cm;
    p.g = 36.7 * mm2cm;
    p.j = 65.7 * mm2cm;   % foot link (was incorrectly 50 before — fixed here)
    p.k = 49.0 * mm2cm;

    % Ground pivots
    p.O = [0;    0  ];   % crank pivot
    p.A = [7.8;  0  ];   % second ground pivot (O-A = 7.8 cm = 78 mm)
    %
    % NOTE: The separation between O and A matters for assembly.
    % Using 78 mm (7.8 cm) from the standard Jansen geometry.
end

function F = jansenFootPoint(theta_k, p)
% Solve one crank angle and return the foot point [x;y].

    O = p.O;
    A = p.A;

    % Crank endpoint B
    B = O + p.a * [cos(theta_k); sin(theta_k)];

    % C: on circle (A, b) and circle (B, c)
    Cc = circleIntersectionsJ(A, p.b, B, p.c);
    C  = pickByJ(Cc, "maxY");

    % D: on circle (A, d) and circle (C, e)
    Dc = circleIntersectionsJ(A, p.d, C, p.e);
    D  = pickByJ(Dc, "minY");

    % E: on circle (B, f) and circle (D, g)
    Ec = circleIntersectionsJ(B, p.f, D, p.g);
    E  = pickByJ(Ec, "minY");

    % Foot F: on circle (C, j) and circle (E, k)
    Fc = circleIntersectionsJ(C, p.j, E, p.k);
    F  = pickByJ(Fc, "minY");

    F_out = F;   %#ok<NASGU>
    F = F;       % return value
end

function Pfoot = jansenFootTrajectory(theta, p)
% Retained for compatibility — wraps the per-angle solver.
    Pfoot = nan(2, numel(theta));
    for k = 1:numel(theta)
        try
            Pfoot(:,k) = jansenFootPoint(theta(k), p);
        catch
            % NaN left in place
        end
    end
end

function pts = circleIntersectionsJ(c1, r1, c2, r2)
% Private version for classic Jansen (avoids name clash with main mechanism).
    dvec = c2 - c1;
    d    = norm(dvec);
    if d < 1e-12
        error("Coincident centers.");
    end
    if d > r1 + r2 + 1e-9 || d < abs(r1 - r2) - 1e-9
        error("Non-assemblable.");
    end
    % clamp for numerical safety
    cosA = (r1^2 + d^2 - r2^2) / (2*r1*d);
    cosA = max(-1, min(1, cosA));
    a    = r1 * cosA;
    h    = r1 * sqrt(max(1 - cosA^2, 0));
    ex   = dvec / d;
    ey   = [-ex(2); ex(1)];
    pm   = c1 + a*ex;
    pts  = [pm + h*ey,  pm - h*ey];
end

function p = pickByJ(pts, mode)
    switch mode
        case "maxY", [~,idx] = max(pts(2,:));
        case "minY", [~,idx] = min(pts(2,:));
        case "maxX", [~,idx] = max(pts(1,:));
        case "minX", [~,idx] = min(pts(1,:));
        otherwise,   error("Unknown mode.");
    end
    p = pts(:,idx);
end