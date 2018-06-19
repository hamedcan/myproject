import numpy as np


def complexity(gt):
    a = 0

    for i in range(0, gt.shape[0]):
        for j in range(0, gt.shape[1]):
            for k in range(0, gt.shape[2]):
                if gt[i, j, k] == 1 and (
                                                gt[i + 1, j, k] < 1 or gt[i - 1, j, k] < 1 or
                                            gt[i, j + 1, k] < 1 or gt[i, j - 1, k] < 1 or gt[i, j, k + 1] < 1 or gt[i, j, k - 1] < 1):
                    a += 1

    v = np.count_nonzero(gt[:, :, :] > 0)

    return float(a ** 3) / float(v ** 2)


gt = np.zeros((5, 5, 5))
pred = np.zeros((5, 5, 5))
pred[2,2,2] = 1
pred[2,2,3] = 1
pred[2,2,1] = 1

gt[2,2,2] = 1
gt[2,2,0] = 1
gt[2,2,1] = 1

m = 1  # margin

x = gt.shape[0]
y = gt.shape[1]
z = gt.shape[2]

margin_pred = np.around(pred[:, :, :])
margin_pred[m:x - m, m:y - m, m:z - m] = np.zeros((x - 2 * m, y - 2 * m, z - 2 * m))

margin_gt = np.around(gt[:, :, :])
margin_gt[m:x - m, m:y - m, m:z - m] = np.zeros((x - 2 * m, y - 2 * m, z - 2 * m))

tp = np.count_nonzero(np.multiply(gt[:, :, :], np.around(pred[:, :, :])))  # AND
tn = np.count_nonzero(np.add(gt[:, :, :], np.around(pred[:, :, :])) == 0)
fp = np.count_nonzero(np.bitwise_and(gt[:, :, :] == 0, pred[:, :, :] == 1))
fn = np.count_nonzero(np.bitwise_and(gt[:, :, :] == 1, pred[:, :, :] == 0))

comp = complexity(gt)
dice = (2 * tp) / ((fp + fn) + 2 * tp)

print("tp " + str(tp))
print("tn " + str(tn))
print("fn " + str(fn))
print("fp " + str(fp))
print("margin pred count " + str(np.count_nonzero(margin_pred)))
print("margin gt count " + str(np.count_nonzero(margin_gt)))
print("complexity " + str(comp))
print("dice " + str(dice))
