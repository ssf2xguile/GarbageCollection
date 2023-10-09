def selection_sort(n, a):
    for i in range(0, n-1):
        for t in range(0, n):
            print(" %d " % a[t], end="")
        print()
        m = a[i]
        k = i
        for j in range(i+1, n):
            if (a[j] < m):
                m = a[j]
                k = j
        a[k] = a[i]
        a[i] = m

if __name__ == "__main__":
    a = [3, 5, 2, 1, 4]
    n = len(a)
    selection_sort(n, a)