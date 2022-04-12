import sys

THRESH = 400
'''Remember this is old version of expt
Value lower than THRESH is a 1 and greater
is a 0.
TODO: Remember to change decode logic!
'''
def decode(f_path):
    msgs = []
    msg = ""
    # We have 100 files, 101 exclusive!
    with open(f_path, "r") as f:
        for line in f.readlines ():
            try:
                if '====' in line:
                    msgs.append(msg)
                    msg = ""
                    continue

                vals = line.split(" ")
                l = len(vals)
                if l == 5:
                    # update errors observed here
                    # Get bit position and supposed_val
                    time = float(vals[3].strip())
                    if time > THRESH:
                        msg += "0"
                    else:
                        msg += "1"
                elif l == 0:
                    continue
            except Exception as e:
                # print(str(e))
                pass

    return msgs


if __name__ == "__main__":
    f_path = sys.argv[1]
    msgs = decode(f_path)
    print(msgs)
