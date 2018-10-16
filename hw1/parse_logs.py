point2vote = {}
for sigma_s in [1,2,3]:
    for sigma_r in [0.05, 0.1, 0.2]:
        point2error = {}
        with open("0c/log_sigmaS{}_sigmaR{}.txt".format(sigma_s, sigma_r)) as myfile:
            for idx, line in enumerate(myfile.readlines()):
                tokens = line.split()
                point2error[(float(tokens[1]), float(tokens[3]), float(tokens[5]))] = float(tokens[8])
#         print(point2error)
        for a in range(11): # 0~10
            for b in range(11-a):
                w_b = a/10
                w_g = b/10
                w_r = round(1 - w_b - w_g, 2)
                
                if round(w_b-0.1,2) >= 0 and round(w_g+0.1,2) <= 1 and point2error[(round(w_b-0.1,2), round(w_g+0.1,2), w_r)] < point2error[(w_b, w_g, w_r)]:
                    continue
                if round(w_b-0.1,2) >= 0 and round(w_r+0.1,2) <= 1 and point2error[(round(w_b-0.1,2), w_g, round(w_r+0.1,2))] < point2error[(w_b, w_g, w_r)]:
                    continue
                if round(w_g-0.1,2) >= 0 and round(w_b+0.1,2) <= 1 and point2error[(round(w_b+0.1,2), round(w_g-0.1,2), w_r)] < point2error[(w_b, w_g, w_r)]:
                    continue
                if round(w_g-0.1,2) >= 0 and round(w_r+0.1,2) <= 1 and point2error[(w_b, round(w_g-0.1,2), round(w_r+0.1,2))] < point2error[(w_b, w_g, w_r)]:
                    continue
                if round(w_r-0.1,2) >= 0 and round(w_g+0.1,2) <= 1 and point2error[(w_b, round(w_g+0.1,2), round(w_r-0.1,2))] < point2error[(w_b, w_g, w_r)]:
                    continue
                if round(w_r-0.1,2) >= 0 and round(w_b+0.1,2) <= 1 and point2error[(round(w_b+0.1,2), w_g, round(w_r-0.1,2))] < point2error[(w_b, w_g, w_r)]:
                    continue
                
                # if got to here, is local minima
                if (w_b, w_g, w_r) not in point2vote:
                    point2vote[(w_b, w_g, w_r)] = 1
                else:
                    point2vote[(w_b, w_g, w_r)] += 1
print("ALL DONE!")
print(point2vote)