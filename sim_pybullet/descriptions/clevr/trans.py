if __name__=='__main__':
    for id in ['cube', 'aux', 'cylinder']:
        with open('clevr_{}-ini.obj'.format(id)) as f_in:
            with open('clevr_{}.obj'.format(id), 'w') as f_out:
                context = f_in.read().split('\n')
                for x in context:
                    if x.startswith('vn'):
                        l = x.split(' ')
                        l[1] = float(l[1]) * (0.02 if id == 'aux' else 0.02)
                        l[2] = float(l[2]) * 0.02
                        l[3] = float(l[3]) * 0.02
                        print('vn {} {} {}'.format(l[1], l[2], l[3]), file=f_out)
                    elif x.startswith('v'):
                        l = x.split(' ')
                        l[1] = float(l[1]) * (0.02 if id == 'aux' else 0.02)
                        l[2] = float(l[2]) * 0.02
                        l[3] = float(l[3]) * 0.02
                        print('v {} {} {}'.format(l[1], l[2], l[3]), file=f_out)
                    else:
                        print(x, file=f_out)

    for lx in [2, 3]:
        for ly in [2, 3]:
            with open('clevr_cube-ini.obj') as f_in:
                with open('clevr_cube{}{}.obj'.format(lx, ly), 'w') as f_out:
                    context = f_in.read().split('\n')
                    for x in context:
                        if x.startswith('vn'):
                            l = x.split(' ')
                            l[1] = float(l[1]) * lx / 100
                            l[2] = float(l[2]) * ly / 100
                            l[3] = float(l[3]) * 0.025
                            print('vn {} {} {}'.format(l[1], l[2], l[3]), file=f_out)
                        elif x.startswith('v'):
                            l = x.split(' ')
                            l[1] = float(l[1]) * lx / 100
                            l[2] = float(l[2]) * ly / 100
                            l[3] = float(l[3]) * 0.025
                            print('v {} {} {}'.format(l[1], l[2], l[3]), file=f_out)
                        else:
                            print(x, file=f_out)