def sort(n, a):
    for j in range(n-1):
        for k in range(n):
            print("%d " % a[k], end='')
        print()
        for i in range(n-j-1):
            if a[i] > a[i+1]:  # 正しい比較を行うために不等号を変更
                tmp = a[i]
                a[i] = a[i+1]
                a[i+1] = tmp

if __name__ == "__main__":
    a = [5,1, 3, 4, 2]
    sort(5, a)