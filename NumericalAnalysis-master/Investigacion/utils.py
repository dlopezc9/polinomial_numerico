def tovalidxy(y, marks):
    """
    Returns two arrays: x and y, where only points marked as True
    are considered.
    """
    #n = marks.count(True)
    n = len(marks)

    validx = [None] * n
    validy = [None] * n

    cnt = 0
    #for i in range(len(y)):
    #    if marks[i]:
    #        validx[cnt] = i
    #        validy[cnt] = y[i]
    #        cnt += 1
    for i in range(len(y)):
        validx[cnt] = i
        validy[cnt] = y[i]
        cnt += 1
        
    return validx, validy

