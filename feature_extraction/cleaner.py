import os

data_path = '../ABI_data/'
llds_path_base = data_path + 'LLDs/'
transcript_path_base = data_path + 'Transcripts/'

with open(data_path +'transcribing_failed.txt', 'r') as f:
    lines = f.readlines()
    failed = []
    for line in lines:
        failed.append(line.strip('\n'))

failed = set(failed)
failed = list(failed)
print('Total %d items' % len(failed))
lld_counter = 0
t_counter = 0

for item in failed:
    podcast = item.split('_')[0]
    llds_path = llds_path_base + podcast + '/' + item + '.csv'
    transcript_path = transcript_path_base + podcast + '/' + item + 'txt'
    if os.path.exists(llds_path):
        lld_counter += 1
        os.remove(llds_path)


    if os.path.exists(transcript_path):
        t_counter+=1
        os.remove(transcript_path)


print('LLDs: %d' % lld_counter)
print('Transcripts: %d' % t_counter)