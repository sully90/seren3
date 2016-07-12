def ncpu():
    '''
    Returns the number of physical CPUs on this machine
    '''
    import commands
    return int(commands.getoutput("grep -c processor /proc/cpuinfo"))

def qlogin(tasks, queue="inf_amd.q", mpi=False):
    '''
    Requests an interavtive session to execute a function
    '''
    import numpy as np
    from subprocess import Popen, PIPE, call, check_output
    import sys, os
    QSUB_TEMPLATE = os.environ.get("SEREN2_INTERACTIVE_QLOGIN", "qlogin -q %s" % queue)

    print 'Submitting qlogin request: %s' % QSUB_TEMPLATE

    def readwhile(stream,func):
        while True:
            line = stream.readline()
            if line!='':
                print line[:-1]
                if func(line): break
            else:
                raise exceptions.Exception("Disconnected unexpectedly.")

    print("Requesting an interactive node")
    pqsub=Popen(['ssh','-t','-t','-4','localhost'],stdin=PIPE,stdout=PIPE,stderr=PIPE)
    pqsub.stdin.write(QSUB_TEMPLATE+"\n")
    pqsub.stdin.write('echo HOSTNAME=`hostname`\n')

    def gethostname(line):
        global hostname
        if line.startswith('HOSTNAME'):
            hostname = line.split('=')[1].strip()
            return True

    print("Waiting for the job to start...")
    readwhile(pqsub.stdout, gethostname)

    pqsub.stdin.write('cd $PBS_O_WORKDIR\n')
    pqsub.stdin.write('echo CD\n')

    raise Exception("Not working - need to write to stdin then readwhile")
    try:
        for task in tasks:
            if mpi:
                NSLOTS = np.round(int(ncpu())/2)
                mpi_task = "mpirun -np {NSLOTS} {EXE}".format(NSLOTS=NSLOTS, EXE=task)
                print mpi_task
                check_output(mpi_task, shell=True)
            else:
                check_output(task, shell=True)
    finally:
        # Kill the session
        pqsub.kill()
        print("Caught exception, cleaned up connections.")

    # Kill the session
    pqsub.kill()
    print("Succesfully cleaned up connections.")
    return True
