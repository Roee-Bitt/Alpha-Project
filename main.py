import whisper
from WER import wer, has_number
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import statistics
import librosa.display


# Checks if the string is a name of a valid 'wav' file
# returns true if string contains '.wav' twice
# (regulatory compliance of dataset)
def valid_wav_file(string):
    string = string.lower()
    ind = string.find(".wav")
    string = string[ind + 4:]
    ind = string.find(".wav")
    return ind != -1


# Checks if the string is a name of a 'txt' file
# returns true if string contains '.txt'
def valid_txt_file(string):
    string = string.lower()
    ind = string.find(".txt")
    return ind != -1


# Cuts off the file extension part ('example.txt' --> 'example')
# returns the name of the file without its extension
def cut_file_ext(string):
    ind = string.find(".")
    string = string[:ind]
    return string


# Formats text according to specific format used to compare strings
# returns formatted string
# gets rid of irrelevant numbers at head of line ('txt' files in dataset contains numbers at start of line)
# gets rid of spaces at head of line (to start at the first letter, for comparing later)
# gets rid of other lines (only first line returned)
def format_txt(string):
    while has_number(string):
        string = string[1:]  # get rid of numbers in file name
    while string[0] == " ":
        string = string[1:]  # get rid of space between number and file name
    ind = string.find("\n")  # get rid of other lines
    if ind == -1:
        return string
    string = string[:ind]
    return string


# Rounds to three digits after the decimal point ('8.1899372' --> '8.190')
# returns x rounded to three decimal digits
def three_decimal(x, pos):
    return '%.3f' % x
    

# Finds the name of the directory searched currently ('C:/Windows:/System32' --> 'System32')
# returns the name of the sub-directory
def curr_sub_dir(string):
    ind = string.find("/")
    while(ind != -1):   # while there are '/' (indicating still not at submost-directory)
        string = string[ind+1:]
    return string

# base root for audio and text files (dataset location on computer)
root = '/home/DATA/TRAIN/DR'

# base root for saving histograms of each speaker
figsave_root = '/home/RESULTS/Histogram-WER-Per-SPKR/DR'

# creating array of addresses for each dialect in dataset
addresses = [root+str(f) for f in range(1, 9)]

avg_wer = []
wer_res = []
spkr_name = []
dialects = ['New England', 'Northern', 'North Midland', 'South Midland', 'Southern', 'New York City', 'Western', 'Army Brat']

# load 'Whisper' model ('base.en' version)
model = whisper.load_model("base.en")

for address in addresses: # for every dialect
    for subdir, dirs, files in os.walk(address): # for every spkr
        spkr_name.append(curr_sub_dir(subdir)) # add its name to array
        for file in files:
            if valid_wav_file(file): # if it's an audio file
                # Load the audio file
                y, sr = librosa.load(os.path.join(subdir, file))

                # Calculate and plot spectrogram
                D = librosa.stft(y)
                S_db = librosa.amplitude_to_db(abs(D), ref=np.max)
                plt.figure(figsize=(10, 5))
                librosa.display.specshow(S_db, x_axis='time', y_axis='log')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'Spectrogram {curr_sub_dir(subdir)} {file}')
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.show()
                #print(os.path.join(subdir, file)) # for validation
                wav_loc = os.path.join(subdir, file)
                result = model.transcribe(wav_loc, fp16=False) # transcribing
                print("System Result: \"" + format_txt(result["text"]) + "\"") # printing result
                for subfile in files:
                    if valid_txt_file(subfile): # if it's a text file
                        txt_loc = os.path.join(subdir, subfile)
                        if cut_file_ext(subfile) == cut_file_ext(file):
                            wer_res.append(wer(txt_loc, result["text"])) # calculating and adding the WER result to array
                            print("Expected: \"" + format_txt(open(txt_loc, 'r').read()) + "\"")
                            print("WER: " + str(wer(txt_loc, result["text"]))) # prints WER result
        ### FOR EACH SPEAKER BY ITSELF ###
        plt.hist(wer_res, edgecolor='black')
        plt.xlabel('WER')
        plt.ylabel('Number of Sentences')
        plt.title(f'Histogram of WER Frequency in Sentences ({dialects[addresses.index(address)]}, {curr_sub_dir(subdir)})')
        plt.grid(True)
        formatter = FuncFormatter(three_decimal)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(formatter)
        print('SAVED: ' +figsave_root + str(addresses.index(address)+1) + curr_sub_dir(subdir) + '.png')
        plt.savefig(figsave_root + str(addresses.index(address)+1) + '/' + curr_sub_dir(subdir) + '.png')
        plt.show()
        wer_res.clear()
    
    if wer_res:
        ### FOR AVG-ING EACH DIALECT ###
        avg_wer.append(statistics.mean(wer_res))
        #wer_res.clear() 
    else:
        print("wer_res is empty")
    
    ### FOR EACH DIALECT, EVERY SPEAKER WER ###
    spkr_name.remove(f'DR{addresses.index(address)+5}')
    plt.xticks(rotation=75)
    bar_plot = plt.bar(range(len(avg_wer)), avg_wer, tick_label=spkr_name, edgecolor='black', width=0.4)
    for bar in bar_plot:
        y_val = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, y_val, round(y_val, 3), va='bottom', ha='center')
    plt.xlabel('Speaker Folder Name')
    plt.ylabel('Average WER')
    plt.title(f'Bar Plot of WER Averages Across Speakers ({dialects[addresses.index(address)]})')
    plt.grid(True)
    plt.show()
    
    ### FOR FREQUENCY OF WER AVERAGES OF SPKRS IN EACH DIALECT ###
    plt.hist([round(x,5) for x in avg_wer], edgecolor='black')
    plt.xlabel('Average WER')
    plt.ylabel('Number of Instances (Speakers)')
    plt.title(f'Histogram of WER Averages Frequency Across Speakers ({dialects[addresses.index(address)]})')
    plt.grid(True)
    plt.show()
    
    ### FOR FREQUENCY OF WER RESULTS IN SENTENCES IN EACH DIALECT (NOT SPKR-SPECIFIC) ###
    plt.hist(wer_res, edgecolor='black')
    plt.xlabel('WER')
    plt.ylabel('Number of Sentences')
    plt.title(f'Histogram of WER Frequency in Sentences ({dialects[addresses.index(address)]})')
    formatter = FuncFormatter(three_decimal)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(formatter)
    plt.grid(True)
    plt.show()
    
    spkr_name.clear() # clearing the spkrs from before
    wer_res.clear() # clearing the WER results from this round

### FOR AVERAGE WER FOR EACH DIALECT, FROM LOW TO HIGH ###
# sorting the arrays together so the dialects matches their results, low to high
for j in range(0, 6):
    for i in range(0, 7):
        if avg_wer[i] > avg_wer[i+1]:
            temp = avg_wer[i]
            avg_wer[i] = avg_wer[i+1]
            avg_wer[i+1] = temp
            tempD = dialects[i]
            dialects[i] = dialects[i+1]
            dialects[i+1] = tempD

bar_plot = plt.bar(range(len(avg_wer)), avg_wer, tick_label=dialects, edgecolor='black', width=0.4)
for bar in bar_plot:
    y_val = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, y_val, round(y_val, 3), va='bottom', ha='center')
plt.xlabel('Dialect')
plt.ylabel('Average WER')
plt.title('Histogram of WER Averages Across American Dialects')
plt.grid(True)
plt.show()