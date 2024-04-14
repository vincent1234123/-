def print_windows(img,windows,line_width=5):

    for i  in range(windows.shape[0]):
        window=list(windows[i]-1)
        if len(window)==5:
            y_1,x_1,y_2,x_2,_=window
        elif len(window)==4:
            y_1, x_1, y_2, x_2 = window
        else:
            print("WRONG")
        # print(x_1.shape)
        for i in range(int(x_1),int(x_2)):
            for j in range(1,line_width+1):
                img[i,int(y_1)+j]=255
            for j in range(1,line_width+1):
                img[i, int(y_2) - j] = 255
        for i in range(int(y_1),int(y_2)):
            for j in range(1,line_width+1):
                img[int(x_1)+j, i] = 255
            for j in range(1,line_width+1):
                img[int(x_2) - j, i] = 255
    return img