import numpy as np
def fc_data():
    tax = np.genfromtxt('C:/Users/Juan/Documents/17-18_Stanford/cs231n/Project/15zpallagi.csv', delimiter=',')
    hpi = np.genfromtxt('C:/Users/Juan/Documents/17-18_Stanford/cs231n/Project/HPI.csv', delimiter=',')
    result = np.zeros((int(tax.shape[0] / 6), tax.shape[1] * 6 - 4))
    # temporal[0,:tax.shape[1]] = tax[0,:]
    zipcodes = tax[:, 0]
    tax = tax[:, 1:]
    for i in range(int(tax.shape[0] / 6)):
        for j in range(6):
            if (j == 0):
                result[i, 0] = zipcodes[i * 6]
                result[i, 1 + (j) * tax.shape[1]:((j) + 1) * tax.shape[1] + 1] = tax[j + i * 6, :]
            else:
                # print((j) * tax.shape[1])
                # print(((j) + 1) * tax.shape[1])
                result[i, (j) * tax.shape[1] + 1:((j) + 1) * tax.shape[1] + 1] = tax[j + i * 6, :]
    filled = False
    for i in range(result.shape[0]):
        for j in range(hpi.shape[0]):
            if (result[i, 0] == hpi[j, 0]):
                #print(result[i, result.shape[1]-1])
                result[i, result.shape[1]-1] = hpi[j, 1]
                filled = True
        if(not filled):
            result[i, result.shape[1] - 1] = np.nan
        filled = False
    print('saving')
    np.save('tabular_data/add_num_data.npy',result)

    