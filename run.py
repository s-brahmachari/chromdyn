from utilities import Run
import sys

n = int(sys.argv[1])

run = Run()

if n==0:
    run.harmtrap()
elif n==1:
    run.harmtrap_with_self_avoidance()
elif n==2:
    run.bad_solvent_with_self_avoidance()
elif n==3:
    run.rouse()
elif n==4:
    run.saw()