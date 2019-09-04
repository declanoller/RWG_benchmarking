import pprint as pp


'''Environment
N_inputs
N_outputs
N_weights,
0 HL
N_weights,
1 HL, 2 HU
N_weights,
1 HL, 4 HU
N_weights,
1 HL, 8 HU
N_weights,
2 HL, 4 HU'''
d = {}
d['CartPole-v0'] = {
    'N_inputs' : 4,
    'N_outputs' : 2,
    '0HL' : '10 (8)',
    '1HL_2HU' : '16 (12)',
    '1HL_4HU' : '30 (24)',
    '1HL_8HU' : '58 (48)',
    '2HL_4HU' : '50 (40)'
}
d['MountainCar-v0'] = {
    'N_inputs' : 2,
    'N_outputs' : 3,
    '0HL' : '9 (6)',
    '1HL_2HU' : '15 (10)',
    '1HL_4HU' : '27 (20)',
    '1HL_8HU' : '51 (40)',
    '2HL_4HU' : '47 (36)'
}
d['MountainCarContinuous-v0'] = {
    'N_inputs' : 2,
    'N_outputs' : 1,
    '0HL' : '3 (2)',
    '1HL_2HU' : '9 (6)',
    '1HL_4HU' : '17 (12)',
    '1HL_8HU' : '33 (24)',
    '2HL_4HU' : '37 (28)'
}
d['Pendulum-v0'] = {
    'N_inputs' : 3,
    'N_outputs' : 1,
    '0HL' : '4 (3)',
    '1HL_2HU' : '11 (8)',
    '1HL_4HU' : '21 (16)',
    '1HL_8HU' : '41 (32)',
    '2HL_4HU' : '41 (32)'
}
d['Acrobot-v1'] = {
    'N_inputs' : 6,
    'N_outputs' : 3,
    '0HL' : '21 (18)',
    '1HL_2HU' : '23 (18)',
    '1HL_4HU' : '43 (36)',
    '1HL_8HU' : '83 (72)',
    '2HL_4HU' : '63 (52)'
}

for k, v in d.items():
    v['env'] = k

d_nobias = {}
for env, env_dict in d.items():
    d_nobias[env] = env_dict.copy()
    for k, v in d_nobias[env].items():
        no_bias_val = str(v)
        # Takes the term inside the parentheses for each.
        if ' ' in no_bias_val:
            no_bias_val = v.split(' ')[1].replace('(', '').replace(')', '')

        d_nobias[env][k] = no_bias_val


d_with_bias = {}
for env, env_dict in d.items():
    d_with_bias[env] = env_dict.copy()
    for k, v in d_with_bias[env].items():
        no_bias_val = str(v)
        # Takes the term inside the parentheses for each.
        if ' ' in no_bias_val:
            no_bias_val = v.split(' ')[0]

        d_with_bias[env][k] = no_bias_val


#pp.pprint(d_nobias)



'''
\begin{center}
 \begin{tabular}{||c c c c||}
 \hline
 Col1 & Col2 & Col2 & Col3 \\
 \hline\hline
 1 & 6 & 87837 & 787 \\
 \hline
 2 & 7 & 78 & 5415 \\
 \hline
 3 & 545 & 778 & 7507 \\
 \hline
 4 & 545 & 18744 & 7560 \\
 \hline
 5 & 88 & 788 & 6344 \\ [1ex]
 \hline
\end{tabular}
\end{center}
'''

col_names_dict = {
    'Environment' : 'env',
    '$N_{inputs}$' : 'N_inputs',
    '$N_{outputs}$' : 'N_outputs',
    '0 HL' : '0HL',
    '1 HL, 2 HU' : '1HL_2HU',
    '1 HL, 4 HU' : '1HL_4HU',
    '1 HL, 8 HU' : '1HL_8HU',
    '2 HL, 4 HU' : '2HL_4HU'
}

table_string = ''

table_string += r'''
\begin{center}
 \begin{tabular}{'''

#table_string += ' '.join(['c']*len(col_names_dict.keys()))
table_string += ' '.join(['p{3cm}']*len(col_names_dict.keys()))

table_string += r'''}
\hline
'''

table_string += (' & '.join([k for k in col_names_dict.keys()]) + r' \\')
table_string += r'''
\hline\hline
'''

table_dict = d_with_bias
for env, env_dict in table_dict.items():

    col_val_list = [env_dict[v] for k,v in col_names_dict.items()]
    table_string += (' & '.join(col_val_list) + r' \\')
    table_string += r'''
    \hline
    '''


table_string += r'''
\end{tabular}
\end{center}
'''

print(table_string)
exit()



print(table_string)


#
