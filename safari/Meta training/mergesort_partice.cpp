#include <iostream>

void merge(int* a, int lo, int mid, int hi, int n) {
  int i = lo, j = mid + 1;
  int* aux = new int[n];
  for (int k = lo; k <= hi; k++) {
    aux[k] = a[k];
  }
  for (int k = lo; k <= hi; k++) {
    if      (i > mid)          a[k] = aux[j++];
    else if (j > hi)           a[k] = aux[i++];
    else if (aux[j] < aux[i])  a[k] = aux[j++];
    else                       a[k] = aux[i++];

    for (int i = 0; i < n; i++){
      std::cout << a[i] << " ";
    }
    std::cout << "\n";
  }
  delete[] aux;
}


int main() {
  int n = 10;
  int* a = new int[n] {1, 4, 6, 6, 8, 2, 4, 5, 9, 9};

  merge(a, 0, 4, 9, n);

  delete[] a;
  return 0;
}