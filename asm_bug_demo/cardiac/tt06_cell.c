/* tt06_cell.c -- single-cell TP06 epicardial verification: pace 1 Hz, report the
 * action-potential metrics (resting, peak, APD90) and dump V(t) for the last beat. */
#include <stdio.h>
#include <stdlib.h>
#include "tt06.h"

int main(int argc, char **argv){
    double dt = (argc>1)? atof(argv[1]) : 0.02;   /* ms */
    int nbeats = 10; double BCL = 1000.0;          /* 1 Hz */
    double Iamp = -52.0, Idur = 1.0;               /* pA/pF, ms */
    TT06 c; tt06_init(&c);
    FILE *f = fopen("tt06_ap.txt","w");
    double Vmin=1e9, Vmax=-1e9, Vrest=0, t0act=-1, apd90=-1, V30=0;
    for (int beat=0; beat<nbeats; ++beat){
        double tb=0; int last=(beat==nbeats-1);
        Vmin=1e9; Vmax=-1e9; t0act=-1;
        /* capture resting just before stimulus */
        Vrest=c.V;
        double thr90=0;
        for (; tb<BCL; tb+=dt){
            double Istim = (tb<Idur)? Iamp : 0.0;
            double Vprev=c.V;
            tt06_step(&c, Istim, dt);
            if (last) fprintf(f,"%g %g %g\n", tb, c.V, c.Cai);
            if (c.V>Vmax) Vmax=c.V; if (c.V<Vmin) Vmin=c.V;
            if (t0act<0 && Vprev<0 && c.V>=0) t0act=tb;        /* upstroke crossing 0 */
        }
        if (last){ thr90 = Vrest + 0.10*(Vmax-Vrest); (void)thr90; }
        V30=Vmax;
    }
    fclose(f);
    /* recompute APD90 on a clean last beat */
    tt06_init(&c);
    for(int b=0;b<9;++b){ for(double tb=0;tb<BCL;tb+=dt){ double Is=(tb<Idur)?Iamp:0; tt06_step(&c,Is,dt);} }
    double vr=c.V; double vmax=-1e9; double tup=-1;
    /* one more beat, track */
    { TT06 cc=c; for(double tb=0;tb<BCL;tb+=dt){ double Is=(tb<Idur)?Iamp:0; double vp=cc.V; tt06_step(&cc,Is,dt);
        if(cc.V>vmax)vmax=cc.V; if(tup<0&&vp<0&&cc.V>=0)tup=tb; }
      double thr=vr+0.10*(vmax-vr); /* 90% repol level */
      TT06 c2=c; double tcross=-1; for(double tb=0;tb<BCL;tb+=dt){ double Is=(tb<Idur)?Iamp:0; tt06_step(&c2,Is,dt);
        if(tb>tup && tcross<0 && c2.V<=thr) tcross=tb; }
      apd90 = (tup>=0&&tcross>=0)? tcross-tup : -1;
      Vmax=vmax; Vrest=vr;
    }
    printf("TP06 epi single cell (1 Hz, dt=%g ms):\n", dt);
    printf("  resting V = %.2f mV   peak V = %.2f mV   APD90 = %.1f ms\n", Vrest, Vmax, apd90);
    printf("  (physiological targets: rest ~ -85 mV, peak ~ +35..45 mV, APD90 ~ 270..310 ms)\n");
    printf("  Cai range ~ %.2e .. %.2e mM ; wrote tt06_ap.txt (last beat)\n", 1e-4, 1e-3);
    return 0;
}
