
def dump_results():
    
    fir_half = input('Please enter first half: ')
    sec_half = '.log' # input('Please enter second half: ')
    count = int(input('Please enter file num: '))
    file_list = [('%s%d%s' % (fir_half, i+1, sec_half)) for i in range(count)]
    print(file_list)

    res_miou = []
    res_pixacc = []
    for file in file_list:
        with open(file, 'r') as f:
            res = f.read()
            if len(res) < 310:
                print('Skipped [%s] since it\'s too short.' % file)
                res_miou.append('TBD')
                res_pixacc.append('TBD')
                continue
            res = res[-300:].split('\n')
            if 'Performance of last 5 epochs' in res:
                # final_idx = res.index('Performance of last 5 epochs')
                res_miou1 = eval(res[6].split(': ')[-1])
                res_pixacc1 = eval(res[7].split(': ')[-1])
                res_miou2, res_pixacc2 = eval(res[8].split(': ')[-1])
                # print(res_miou1, res_miou2)
                # print(res_pixacc1, res_pixacc1)
                res_miou.append('%.4f / %.4f' % (res_miou1, res_miou2))
                res_pixacc.append('%.4f / %.4f' % (res_pixacc1, res_pixacc2))
            else:
                print('Skipped [%s] since it\'s incomplete.' % file)
                res_miou.append('TBD')
                res_pixacc.append('TBD')
    
    print('[mIoU]:', '\n'.join(res_miou), sep='\n')
    print('\n[Pix_Acc]:', '\n'.join(res_pixacc), sep='\n')
    

if __name__ == '__main__':
    dump_results()