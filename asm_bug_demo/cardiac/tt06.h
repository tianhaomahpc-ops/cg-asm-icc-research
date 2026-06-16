/* tt06.h -- ten Tusscher & Panfilov 2006 (TP06) human ventricular cell, EPICARDIAL.
 * Self-contained single-cell model: Rush-Larsen for gating, forward Euler for
 * concentrations (as in slides/tex/cardiac.pdf). Units: V[mV], t[ms], currents
 * [pA/pF], Ca[mM], Na/K[mM]. Reference: ten Tusscher-Panfilov, AJP Heart 291:H1088
 * (2006); g_* constants match cardioid/elec/BetterTT06.cc (epi). */
#ifndef TT06_H
#define TT06_H
#include <math.h>

/* ---- physical constants ---- */
#define TT_R    8314.472
#define TT_T    310.0
#define TT_F    96485.3415
#define TT_RTONF (TT_R*TT_T/TT_F)          /* ~26.71 mV */
/* external concentrations */
#define TT_Ko   5.4
#define TT_Nao  140.0
#define TT_Cao  2.0
/* cell geometry / capacitance (uF, um^3) */
#define TT_Cm   0.185
#define TT_Vc   0.016404
#define TT_Vsr  0.001094
#define TT_Vss  0.00005468
/* maximal conductances (epi) */
#define TT_GNa  14.838
#define TT_GK1  5.405
#define TT_Gto  0.294
#define TT_GKr  0.153
#define TT_GKs  0.392
#define TT_GCaL 3.98e-5
#define TT_GpCa 0.1238
#define TT_GpK  0.0146
#define TT_GbNa 0.00029
#define TT_GbCa 0.000592
/* pumps / exchangers */
#define TT_PNaK 2.724
#define TT_KmNa 40.0
#define TT_Kmk  1.0
#define TT_kNaCa 1000.0
#define TT_Ksat  0.1
#define TT_alpha 2.5
#define TT_gamma 0.35
#define TT_KmNai 87.5
#define TT_KmCa  1.38
#define TT_KpCa  0.0005
/* SR handling */
#define TT_Vmaxup 0.006375
#define TT_Kup    0.00025
#define TT_Vrel   0.102
#define TT_k1p    0.15
#define TT_k2p    0.045
#define TT_k3     0.06
#define TT_k4     0.005
#define TT_EC     1.5
#define TT_maxsr  2.5
#define TT_minsr  1.0
#define TT_Vleak  0.00036
#define TT_Vxfer  0.0038
/* buffers */
#define TT_Bufc   0.2
#define TT_Kbufc  0.001
#define TT_Bufsr  10.0
#define TT_Kbufsr 0.3
#define TT_Bufss  0.4
#define TT_Kbufss 0.00025

/* 19 state variables */
typedef struct {
    double V, Cai, CaSR, CaSS, Nai, Ki;
    double m, h, j, xr1, xr2, xs, r, s, d, f, f2, fCass, Rbar;
} TT06;

static inline void tt06_init(TT06 *c){
    c->V=-85.23; c->Cai=1.06e-4; c->CaSR=3.55; c->CaSS=2.14e-4; c->Nai=8.604; c->Ki=136.89;
    c->m=1.72e-3; c->h=0.7444; c->j=0.7045; c->xr1=2.42e-4; c->xr2=0.4720; c->xs=8.7e-3;
    c->r=2.42e-8; c->s=0.999998; c->d=3.29e-5; c->f=0.7; c->f2=0.9; c->fCass=0.9999; c->Rbar=0.9893;
}

/* reaction substep: advance gating (RL) + concentrations (FE) by dt using the CURRENT
 * V, return Iion (pA/pF = mV/ms). Does NOT update V (the monodomain PDE does that). */
static inline double tt06_react(TT06 *c, double dt){
    const double V=c->V;
    const double Ek =TT_RTONF*log(TT_Ko/c->Ki);
    const double Ena=TT_RTONF*log(TT_Nao/c->Nai);
    const double Eks=TT_RTONF*log((TT_Ko+0.03*TT_Nao)/(c->Ki+0.03*c->Nai));
    const double Eca=0.5*TT_RTONF*log(TT_Cao/c->Cai);

    /* IK1 */
    double ak1=0.1/(1.0+exp(0.06*(V-Ek-200.0)));
    double bk1=(3.0*exp(2e-4*(V-Ek+100.0))+exp(0.1*(V-Ek-10.0)))/(1.0+exp(-0.5*(V-Ek)));
    double IK1=TT_GK1*sqrt(TT_Ko/5.4)*(ak1/(ak1+bk1))*(V-Ek);
    /* Ito */
    double Ito=TT_Gto*c->r*c->s*(V-Ek);
    /* IKr */
    double IKr=TT_GKr*sqrt(TT_Ko/5.4)*c->xr1*c->xr2*(V-Ek);
    /* IKs */
    double IKs=TT_GKs*c->xs*c->xs*(V-Eks);
    /* ICaL */
    double ICaL=TT_GCaL*c->d*c->f*c->f2*c->fCass*4.0*(V-15.0)*(TT_F/TT_RTONF)*
        (0.25*c->CaSS*exp(2.0*(V-15.0)/TT_RTONF)-TT_Cao)/(exp(2.0*(V-15.0)/TT_RTONF)-1.0);
    /* INa */
    double INa=TT_GNa*c->m*c->m*c->m*c->h*c->j*(V-Ena);
    /* IbNa, IbCa */
    double IbNa=TT_GbNa*(V-Ena);
    double IbCa=TT_GbCa*(V-Eca);
    /* INaK */
    double INaK=TT_PNaK*(TT_Ko/(TT_Ko+TT_Kmk))*(c->Nai/(c->Nai+TT_KmNa))/
        (1.0+0.1245*exp(-0.1*V/TT_RTONF)+0.0353*exp(-V/TT_RTONF));
    /* INaCa */
    double INaCa=TT_kNaCa*(exp(TT_gamma*V/TT_RTONF)*c->Nai*c->Nai*c->Nai*TT_Cao-
        exp((TT_gamma-1.0)*V/TT_RTONF)*TT_Nao*TT_Nao*TT_Nao*c->Cai*TT_alpha)/
        ((TT_KmNai*TT_KmNai*TT_KmNai+TT_Nao*TT_Nao*TT_Nao)*(TT_KmCa+TT_Cao)*
         (1.0+TT_Ksat*exp((TT_gamma-1.0)*V/TT_RTONF)));
    /* IpCa, IpK */
    double IpCa=TT_GpCa*c->Cai/(c->Cai+TT_KpCa);
    double IpK=TT_GpK*(V-Ek)/(1.0+exp((25.0-V)/5.98));

    double Iion=IK1+Ito+IKr+IKs+ICaL+INa+IbNa+IbCa+INaK+INaCa+IpCa+IpK;

    /* ---- gating (Rush-Larsen): steady-state + tau, exponential update ---- */
    double minf=1.0/((1.0+exp((-56.86-V)/9.03))*(1.0+exp((-56.86-V)/9.03)));
    double am=1.0/(1.0+exp((-60.0-V)/5.0));
    double bm=0.1/(1.0+exp((V+35.0)/5.0))+0.10/(1.0+exp((V-50.0)/200.0));
    double tm=am*bm;
    double hinf=1.0/((1.0+exp((V+71.55)/7.43))*(1.0+exp((V+71.55)/7.43)));
    double ah=(V>=-40.0)?0.0:0.057*exp(-(V+80.0)/6.8);
    double bh=(V>=-40.0)?0.77/(0.13*(1.0+exp(-(V+10.66)/11.1))):2.7*exp(0.079*V)+3.1e5*exp(0.3485*V);
    double th=1.0/(ah+bh);
    double jinf=hinf;
    double aj=(V>=-40.0)?0.0:(-2.5428e4*exp(0.2444*V)-6.948e-6*exp(-0.04391*V))*(V+37.78)/(1.0+exp(0.311*(V+79.23)));
    double bj=(V>=-40.0)?0.6*exp(0.057*V)/(1.0+exp(-0.1*(V+32.0))):0.02424*exp(-0.01052*V)/(1.0+exp(-0.1378*(V+40.14)));
    double tj=1.0/(aj+bj);
    double xr1inf=1.0/(1.0+exp((-26.0-V)/7.0));
    double axr1=450.0/(1.0+exp((-45.0-V)/10.0)); double bxr1=6.0/(1.0+exp((V+30.0)/11.5));
    double txr1=axr1*bxr1;
    double xr2inf=1.0/(1.0+exp((V+88.0)/24.0));
    double axr2=3.0/(1.0+exp((-60.0-V)/20.0)); double bxr2=1.12/(1.0+exp((V-60.0)/20.0));
    double txr2=axr2*bxr2;
    double xsinf=1.0/(1.0+exp((-5.0-V)/14.0));
    double axs=1400.0/sqrt(1.0+exp((5.0-V)/6.0)); double bxs=1.0/(1.0+exp((V-35.0)/15.0));
    double txs=axs*bxs+80.0;
    double rinf=1.0/(1.0+exp((20.0-V)/6.0)); double tr=9.5*exp(-(V+40.0)*(V+40.0)/1800.0)+0.8;
    double sinf=1.0/(1.0+exp((V+20.0)/5.0));               /* epi/M */
    double ts=85.0*exp(-(V+45.0)*(V+45.0)/320.0)+5.0/(1.0+exp((V-20.0)/5.0))+3.0;
    double dinf=1.0/(1.0+exp((-8.0-V)/7.5));
    double ad=1.4/(1.0+exp((-35.0-V)/13.0))+0.25; double bd=1.4/(1.0+exp((V+5.0)/5.0));
    double gd=1.0/(1.0+exp((50.0-V)/20.0)); double td=ad*bd+gd;
    double finf=1.0/(1.0+exp((V+20.0)/7.0));
    double tf=1102.5*exp(-(V+27.0)*(V+27.0)/225.0)+200.0/(1.0+exp((13.0-V)/10.0))+180.0/(1.0+exp((V+30.0)/10.0))+20.0;
    double f2inf=0.67/(1.0+exp((V+35.0)/7.0))+0.33;
    double tf2=562.0*exp(-(V+27.0)*(V+27.0)/240.0)+31.0/(1.0+exp((25.0-V)/10.0))+80.0/(1.0+exp((V+30.0)/10.0));
    double fCassinf=0.6/(1.0+(c->CaSS/0.05)*(c->CaSS/0.05))+0.4;
    double tfCass=80.0/(1.0+(c->CaSS/0.05)*(c->CaSS/0.05))+2.0;

    #define RL(g,ginf,tau) (g)=(ginf)-((ginf)-(g))*exp(-dt/(tau))
    RL(c->m,minf,tm); RL(c->h,hinf,th); RL(c->j,jinf,tj);
    RL(c->xr1,xr1inf,txr1); RL(c->xr2,xr2inf,txr2); RL(c->xs,xsinf,txs);
    RL(c->r,rinf,tr); RL(c->s,sinf,ts); RL(c->d,dinf,td);
    RL(c->f,finf,tf); RL(c->f2,f2inf,tf2); RL(c->fCass,fCassinf,tfCass);
    #undef RL

    /* ---- Ca dynamics (forward Euler) ---- */
    double kcasr=TT_maxsr-(TT_maxsr-TT_minsr)/(1.0+(TT_EC/c->CaSR)*(TT_EC/c->CaSR));
    double k1=TT_k1p/kcasr, k2=TT_k2p*kcasr;
    double dRbar=-k2*c->CaSS*c->Rbar+TT_k4*(1.0-c->Rbar);
    c->Rbar+=dt*dRbar;
    double O=k1*c->CaSS*c->CaSS*c->Rbar/(TT_k3+k1*c->CaSS*c->CaSS);
    double Irel=TT_Vrel*O*(c->CaSR-c->CaSS);
    double Ileak=TT_Vleak*(c->CaSR-c->Cai);
    double Iup=TT_Vmaxup/(1.0+TT_Kup*TT_Kup/(c->Cai*c->Cai));
    double Ixfer=TT_Vxfer*(c->CaSS-c->Cai);

    double CaSRbuf=1.0/(1.0+TT_Bufsr*TT_Kbufsr/((c->CaSR+TT_Kbufsr)*(c->CaSR+TT_Kbufsr)));
    c->CaSR+=dt*CaSRbuf*(Iup-Irel-Ileak);
    double CaSSbuf=1.0/(1.0+TT_Bufss*TT_Kbufss/((c->CaSS+TT_Kbufss)*(c->CaSS+TT_Kbufss)));
    double dCaSS=(-ICaL*TT_Cm/(2.0*TT_Vss*TT_F))+(Irel*TT_Vsr/TT_Vss)-(Ixfer*TT_Vc/TT_Vss);
    c->CaSS+=dt*CaSSbuf*dCaSS;
    double Caibuf=1.0/(1.0+TT_Bufc*TT_Kbufc/((c->Cai+TT_Kbufc)*(c->Cai+TT_Kbufc)));
    double dCai=Caibuf*((-(IbCa+IpCa-2.0*INaCa)*TT_Cm/(2.0*TT_Vc*TT_F))+(Ileak-Iup)*TT_Vsr/TT_Vc+Ixfer);
    c->Cai+=dt*dCai;

    /* ---- Na, K (forward Euler); stimulus is external charge -> affects V only ---- */
    c->Nai+=dt*(-(INa+IbNa+3.0*INaK+3.0*INaCa)*TT_Cm/(TT_Vc*TT_F));
    c->Ki +=dt*(-(IK1+Ito+IKr+IKs+IpK-2.0*INaK)*TT_Cm/(TT_Vc*TT_F));

    return Iion;
}

/* single-cell convenience: react + advance V by forward Euler (external stim, mV/ms). */
static inline double tt06_step(TT06 *c, double Istim, double dt){
    double Iion = tt06_react(c, dt);
    c->V += dt*(-(Iion+Istim));
    return Iion;
}
#endif
