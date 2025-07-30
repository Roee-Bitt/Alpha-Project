import re

# Custom-programmed function that implements a basic custom-developed WER formula algorithm.
# Takes 2 strings: origin - actual words said in the audio file; output - the ASR system's output
# Returns calculated WER based on the 2 strings

def wer_base(origin, output):
    origin = origin.lower()
    origin = re.sub(r'[^\w\s]', '', origin)
    origin = origin.strip()
    output = output.lower()
    output = re.sub(r'[^\w\s]', '', output)
    output = output.strip()

    origin_arr = origin.split()
    output_arr = output.split()

    count_sub = 0
    count_ins = 0
    count_del = 0
    len_origin = len(origin_arr)
    len_output = len(output_arr)

    indorgn = 0
    indoutp = 0
    
    DELETE_FLAG = 998
    INSERT_FLAG = 999

    k = 0
    j = 0

    while indoutp < len_output and indorgn < len_origin:
        if origin_arr[indorgn] != output_arr[indoutp]:
            last_del = indorgn
            for k in range(1, len_origin):
                if indorgn + k < len_origin:
                    if origin_arr[indorgn + k] == output_arr[indoutp]:
                        indorgn += k
                        k = DELETE_FLAG
                        break
                else:
                    k = -1
                    break
            last_ins = indoutp
            for j in range(1, len_output):
                if indoutp + j < len_output:
                    if origin_arr[indorgn] == output_arr[indoutp + j]:
                        indoutp += j
                        j = INSERT_FLAG
                        break
                else:
                    j = -1
                    break
            if j == INSERT_FLAG:
                count_ins += indoutp - last_ins
            elif k == DELETE_FLAG:
                count_del += indorgn - last_del
            else:
                count_sub += 1
        indorgn += 1
        indoutp += 1

    wer_score = (count_del + count_ins + count_sub) / len_origin
    return wer_score

# Function for WER calculation by using system address of a file containing the actual words said in the audio file.

def wer(origin_address, output):
    origin = open(origin_address, 'r').read()
    while has_number(origin):
        origin = origin[1:]
    return wer_base(origin, output)


def has_number(string):
    for char in string:
        if char.isdigit():
            return True
    return False
