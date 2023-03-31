#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <ctime> // time
#include <map>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template<typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in)
{

  std::vector<double> linspaced;

  double start = static_cast<double>(start_in);
  double end = static_cast<double>(end_in);
  double num = static_cast<double>(num_in);

  if (num == 0) { return linspaced; }
  if (num == 1) 
    {
      linspaced.push_back(start);
      return linspaced;
    }

  double delta = (end - start) / (num - 1);

  for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
  linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input
  return linspaced;
}

template<typename T>
void display_progress(T &ini, T &total, int &carrier, int &sp) {
  int true_percent;
  double percent = static_cast<double>(ini) / static_cast<double>(total);
  percent *= 100;
  percent = floor(percent);
  true_percent = (int) percent;
  if (true_percent % sp == 0 && carrier != true_percent) {
    std::cout << true_percent << "%   ";
    carrier = true_percent;
  }
}

void network_density(
  std::vector<std::vector<int> > &A, int &nodes, double &den
) {
  // Create variables ----
  den = 0;
  for (int i = 0; i < nodes; i++) {
    for (int j = 0; j < nodes; j++) {
      if (A[i][j] > 0)
        den++;
    }
  }
  den /= static_cast<double>(nodes * (nodes - 1));
}

void network_M(
  std::vector<std::vector<int> > &A, int &nrows, int&ncols, int &m
) {
  // Create variables ----
  m = 0;
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      if (A[i][j] > 0)
        m++;
    }
  }
}

void network_count(
  std::vector<std::vector<int> > &A, int &nodes, int &counter
) {
  // Create variables ----
  counter = 0;
  for (int i = 0; i < nodes; i++) {
    for (int j = 0; j < nodes; j++) {
      counter += A[i][j];
    }
  }
}

void network_count_M(
  std::vector<std::vector<int> > &A, int &nrows, int &ncols, int &counter
) {
  // Create variables ----
  counter = 0;
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      counter += A[i][j];
    }
  }
}

void network_links(
  std::vector<std::vector<int> > &A, int &nodes, double &counter
) {
  // Create variables ----
  counter = 0;
  for (int i = 0; i < nodes; i++) {
    for (int j = 0; j < nodes; j++) {
      if (A[i][j] > 0)
        counter++;
    }
  }
}

template<typename T>
void unique(std::vector<T> &v) {
  std::vector<int>::iterator ip;
  std::sort(v.begin(), v.end());
  ip = std::unique(
    v.begin(),
    v.begin() + v.size()
  );
  v.resize(
    std::distance(v.begin(), ip)
  );
}

// elliptic***

std::vector<std::vector<int> > sample_elliptic(
  std::vector<std::vector<double> > &D,
  int &nodes, double &rho, double &lb
) {
  // Initialize quantities ----
  int x1, x2, t = 0, sp = 5;
  double Rho = 0, d, p_d, luck;
  std::vector<std::vector<int> > A(
    nodes, std::vector<int>(nodes, 0)
  );
  //  Start loop ----
  //// Match density ----
  std::cout << "Matching network density:\n";
  ////
  while(Rho <= rho) {
    display_progress(Rho, rho, t, sp);
    x1 = rand() % nodes;
    x2 = rand() % nodes;
    if (x1 == x2) continue;
    d = D[x1][x2];
    if (lb > 0) p_d = lb * exp(- lb * d);
    else if (lb == 0) p_d = 0.5;
    luck = rand() % 10000 / 10000.0;
    if (p_d <= luck) A[x1][x2]++;
    network_density(A, nodes, Rho);
  }
  std::cout << "\nDone!\n";
  return A;
}

std::vector<std::vector<int> > const_sample_elliptic(
  std::vector<std::vector<double> > &D,
  int &nodes, int &leaves,
  double &rho, double &lb
) {
  // Initialize quantities ----
  int x1, x2, Count, t = 0, sp = 5;
  double Rho = 0, d, p_d, luck;
  std::vector<std::vector<int> > A(
    nodes, std::vector<int>(nodes, 0)
  );
  //  Start loop ----
  //// Match density ----
  std::cout << "Matching network density:\n";
  while(Rho <= rho) {
    display_progress(Rho, rho, t, sp);
    x1 = rand() % nodes;
    x2 = rand() % nodes;
    if (x1 == x2)
      continue;
    d = D[x1][x2];
    if (lb > 0) p_d = lb * exp(- lb * d);
    else if (lb == 0) p_d = 0.5;
    luck = rand() % 10000 / 10000.0;
    if (p_d <= luck) continue;
    A[x1][x2]++;
    network_density(A, nodes, Rho);
  }
  std::cout << "\nDone!\n";
  //// Match number of edges ----
  std::cout << "\nMatching count quantity:\n";
  t = 0;
  network_count(A, nodes, Count);
  while (Count <= leaves) {
    display_progress(Count, leaves, t, sp);
    x1 = rand() % nodes;
    x2 = rand() % nodes;
    if (A[x1][x2] == 0)
      continue;
    d = D[x1][x2];
    if (lb > 0) p_d = lb * exp(- lb * d);
    else if (lb == 0) p_d = 1;
    luck = rand() % 10000 / 10000.0;
    if (p_d <= luck) continue;
    A[x1][x2]++;
    Count++;
  }
  std::cout << "\nDone!\n";
  return A;
}

// distbase***

void  get_subnets(
  std::vector<std::vector<double> > &net,
  std::vector<double> &bins,
  int &leaves, int& nbins,
  std::vector<std::vector<std::vector<double> > > &subnet
) {
  // Declare variables ----
  std::vector<std::vector<double> > sub;
  // Start loops ----
  for (int i = 0; i < nbins - 1; i++) {
    for (int j = 0; j < leaves; j++) {
      if (i < nbins - 2) {
        if (bins[i] <= net[j][2] && bins[i + 1] > net[j][2])
          sub.push_back(net[j]);
      } else {
        if (bins[i] <= net[j][2] && bins[i + 1] + 0.1 > net[j][2])
          sub.push_back(net[j]);
      }
      
    }
    subnet.push_back(sub);
    sub.clear();
  }
}

std::vector<std::vector<double> > find_bin(double &d, std::vector<std::vector<double> >  &net) {
  std::vector<std::vector<double> > subnet;
  for (int i = 0; i < net.size(); i++) {
    if (d >= net[i][2] - 1e-2 && d < net[i][2] + 1e-2)
      subnet.push_back(net[i]);  
  }
  return subnet;
}

int find_bin(double &d, std::vector<double> &bins, int & nbins) {
  int c = -1;
  for (int i = 0; i < nbins - 1; i++) {
    if (i < nbins - 2) {
      if (bins[i] <= d && bins[i + 1] > d) {
        c = i;
        return c;
      }
    } else {
      if (bins[i] <= d && bins[i + 1] + 0.1 > d) {
        c = i; 
        return c;
      }
    }
   
  }
  return c;
}

double sample_dist_from_prob(
  std::vector<double> &prob, int &nprob, std::vector<double> x
) {
  double suma = 0, d;
  double luck = (rand() % 1000) / 1000.0;
  for (int i = 0; i < nprob; i++) {
    if (suma >= luck) {
      d = x[i];
      break;
    }
    suma += prob[i];
  }
  return d;
}

std::vector<std::vector<int> > sample_from_prob(
  std::vector<std::vector<double> > &net,
  std::vector<double> &bins, int &nbins,
  std::vector<double> &prob, int &nprob,
  std::vector<double> x,
  int &nodes, int &leaves, double &rho
) {
  // Change random seed ----
  srand(time(0));
  // Declare variables ----
  int c, r, A, B, t = 0, sp = 5;
  double mind = bins[0], maxd = floor(bins[nbins - 1]);
  double Rho = 0.0, d, p_d, luck;
  // Declare network ----
  std::vector<std::vector<int> > NET(
    nodes, std::vector<int>(nodes, 0)
  );
  // Categorize distances ----
  std::vector<std::vector<std::vector<double> > > subnet;
  get_subnets(
    net, bins, leaves, nbins, subnet
  );
  std::cout << "Matching network density:\n";
  while (Rho <= rho) {
    display_progress(Rho, rho, t, sp);
    // Generate random distance
    d = sample_dist_from_prob(prob, nprob, x);
    c = find_bin(d, bins, nbins);
    if (subnet[c].size() == 0) continue;
    r = rand() % subnet[c].size();
    A = rand() % 2;
    if (A == 0) B = 1;
    else B = 0;
    NET[subnet[c][r][A]][subnet[c][r][B]]++;
    network_density(NET, nodes, Rho);
  }
  std::cout << "\nDone!\n";
  return NET;
}

std::vector<std::vector<int> > const_sample_from_prob(
  std::vector<std::vector<double> > &net,
  std::vector<double> &bins, int &nbins, int &nlinks,
  std::vector<double> &prob, int &nprob,
  std::vector<double> x,
  int &nodes, int &leaves, double &rho
) {
  // Change random seed ----
  srand(time(0));
  // Declare variables ----
  int c, r, A, B, t = 0, sp = 5, Count = 0;
  double mind = bins[0], maxd = floor(bins[nbins - 1]);
  double Rho = 0.0, d, p_d, luck;
  // Declare network ----
  std::vector<std::vector<int> > NET(
    nodes, std::vector<int>(nodes, 0)
  );
  // Categorize distances ----
  std::vector<std::vector<std::vector<double> > > subnet;
  get_subnets(
    net, bins, leaves, nbins, subnet
  );
  std::cout << "Matching network density:\n";
  while (Rho <= rho) {
    display_progress(Rho, rho, t, sp);
    // Generate random distance
    d = sample_dist_from_prob(prob, nprob, x);
    c = find_bin(d, bins, nbins);
    if (c == -1) continue;
    if (subnet[c].size() == 0) continue;
    r = rand() % subnet[c].size();
    A = rand() % 2;
    if (A == 0) B = 1;
    else B = 0;
    NET[subnet[c][r][A]][subnet[c][r][B]]++;
    network_density(NET, nodes, Rho);
  }
  std::cout << "\nDone!\n";
  // Start matching count number ----
  std::cout << "\nMatching count quantity:\n";
  t = 0;
  network_count(NET, nodes, Count);
  while (Count <= nlinks) {
    display_progress(Count, nlinks, t, sp);
    // Generate random distance
    d = sample_dist_from_prob(prob, nprob, x);
    c = find_bin(d, bins, nbins);
    if (c == -1) continue;
    if (subnet[c].size() == 0) continue;
    r = rand() % subnet[c].size();
    A = rand() % 2;
    if (A == 0) B = 1;
    else B = 0;
    if (NET[subnet[c][r][A]][subnet[c][r][B]] == 0)
      continue;
    NET[subnet[c][r][A]][subnet[c][r][B]]++;
    network_count(NET, nodes, Count);
  }
  return NET;
}

std::vector<std::vector<int> > sample_pareto(
  std::vector<std::vector<double> > &net,
  std::vector<double> &bins, int &nbins,
  int &nodes, int &ncols, int &leaves,
  double &rho, double &a, double &xm, double &xx
) {
  // Change random seed ----
  srand(time(0));
  // Declare variables ----
  int c, r, A, B, t = 0, sp = 5;
  double Rho = 0.0, d;
  //
  std::vector<std::vector<std::vector<double> > > subnet;
  get_subnets(net, bins, leaves, nbins, subnet);
  // Declare network ----
  std::vector<std::vector<int> > NET(nodes, std::vector<int>(nodes, 0));
  std::cout << "Matching network density:\n";
  while (Rho < rho) {
    display_progress(Rho, rho, t, sp);
    // Generate random distance
    d = (rand() % 1000) / 1000.;
    // std::cout << d << "\t" <<  "HELP\n";
    d = xm * pow(1 - d, -1/a);
    // std::cout << d << "\tNohelper\n";
    if (d > xx || d < xm) continue;
    c = find_bin(d, bins, nbins);
    if (c == -1) continue;
    if (subnet[c].size() == 0) continue;
    r = rand() % subnet[c].size();
    A = rand() % 2;
    if (A == 0) B = 1;
    else B = 0;
    NET[subnet[c][r][A]][subnet[c][r][B]]++;
    network_density(NET, ncols, Rho);
  }
  if (Rho > rho)
     NET[subnet[c][r][A]][subnet[c][r][B]]--;
  std::cout << "\nDone!\n";
  return NET;
}

std::vector<std::vector<int> > sample_pareto_trunc(
  std::vector<std::vector<double> > &net,
  std::vector<double> &bins, int &nbins,
  int &nodes, int &ncols, int &leaves,
  double &rho, double &a, double &xm, double &xx
) {
  // Change random seed ----
  srand(time(0));
  // Declare variables ----
  int c, r, A, B, t = 0, sp = 5;
  double Rho = 0.0, d;
  //
  std::vector<std::vector<std::vector<double> > > subnet;
  get_subnets(net, bins, leaves, nbins, subnet);
  // Declare network ----
  std::vector<std::vector<int> > NET(nodes, std::vector<int>(nodes, 0));
  std::cout << "Matching network density:\n";
  while (Rho < rho) {
    display_progress(Rho, rho, t, sp);
    // Generate random distance
    d = (rand() % 1000) / 1000.;
    // std::cout << d << "\t" <<  "HELP\n";
    d = xm * pow(1 - d * (1 - pow(xm / xx, a)), -1 / a);
    // std::cout << d << "\tNohelper\n";
    if (d > xx || d < xm) continue;
    c = find_bin(d, bins, nbins);
    if (c == -1) continue;
    if (subnet[c].size() == 0) continue;
    r = rand() % subnet[c].size();
    A = rand() % 2;
    if (A == 0) B = 1;
    else B = 0;
    NET[subnet[c][r][A]][subnet[c][r][B]]++;
    network_density(NET, ncols, Rho);
  }
  if (Rho > rho)
     NET[subnet[c][r][A]][subnet[c][r][B]]--;
  std::cout << "\nDone!\n";
  return NET;
}

std::vector<std::vector<int> > const_sample_pareto(
  std::vector<std::vector<double> > &net,
  std::vector<double> &bins, int &nbins,
  int &nodes, int &leaves, int &M,
  double &rho, double &a, double &xm, double &xx,
  int &nrows, int &ncols
) {
  // Change random seed ----
  srand(time(0));
  // Declare variables ----
  int c, r, A, B, t = 0, sp = 5, Count = 0;
  double Rho = 0.0, d;
  //
  std::vector<std::vector<std::vector<double> > > subnet;
  get_subnets(net, bins, leaves, nbins, subnet);
  // Declare network ----
  std::vector<std::vector<int> > NET(nodes, std::vector<int>(nodes, 0));
  std::cout << "Matching network density:\n";
  while (Rho < rho) {
    display_progress(Rho, rho, t, sp);
    // Generate random distance
    d = (rand() % 1000) / 1000.;
    d = xm * pow(1 - d, -1/a);
    if (d > xx || d < xm) continue;
    c = find_bin(d, bins, nbins);
    if (c == -1) continue;
    if (subnet[c].size() == 0) continue;
    r = rand() % subnet[c].size();
    A = rand() % 2;
    if (A == 0) B = 1;
    else B = 0;
    NET[subnet[c][r][A]][subnet[c][r][B]]++;
    network_density(NET, ncols, Rho);
  }
  if (Rho > rho)
     NET[subnet[c][r][A]][subnet[c][r][B]]--;
  // Filing up until reaching the same number of neurons ----
  std::cout << "\nMatching number of neurons:\n";
  t = 0;
  network_count_M(NET, nrows, ncols, Count);
  while (Count <= M) {
    display_progress(Count, M, t, sp);
    // Generate random distance
    d = (rand() % 1000) / 1000.;
    d = xm * pow(1 - d, -1/a);
    if (d > xx || d < xm) continue;
    c = find_bin(d, bins, nbins);
    if (c == -1) continue;
    if (subnet[c].size() == 0) continue;
    r = rand() % subnet[c].size();
    A = rand() % 2;
    if (A == 0) B = 1;
    else B = 0;
    if (NET[subnet[c][r][A]][subnet[c][r][B]] == 0) continue;
    NET[subnet[c][r][A]][subnet[c][r][B]]++;
    if (subnet[c][r][A] < nrows && subnet[c][r][B] < ncols) Count++;
  }
  std::cout << "\nDone!\n";
  return NET;
}

std::vector<std::vector<int> > sample_distbase(
  std::vector<std::vector<double> > &net,
  std::vector<double> &bins, int &nbins,
  int &nodes, int &ncols, int &leaves,
  double &rho, double &lb
) {
  // Change random seed ----
  srand(time(0));
  // Declare variables ----
  int c, r, A, B, t = 0, sp = 5;
  double Rho = 0.0, d;
  // Declare network ----
  std::vector<std::vector<int> > NET(
    nodes, std::vector<int>(nodes, 0)
  );
  // Categorize distances ----
  std::vector<std::vector<std::vector<double> > > subnet;
  get_subnets(
    net, bins, leaves, nbins, subnet
  );
  std::cout << "Matching network density:\n";
  while (Rho < rho) {
    display_progress(Rho, rho, t, sp);
    // Generate random distance
    d =  (rand() % 1000) / 1000.;
    d = - log(1 - d) / lb ;
    if (d > bins[nbins - 1]) continue;
    c = find_bin(d, bins, nbins);
    if (c == -1) continue;
    if (subnet[c].size() == 0) continue;
    r = rand() % subnet[c].size();
    A = rand() % 2;
    if (A == 0) B = 1;
    else B = 0;
    NET[subnet[c][r][A]][subnet[c][r][B]]++;
    network_density(NET, ncols, Rho);
  }
  if (Rho > rho)
     NET[subnet[c][r][A]][subnet[c][r][B]]--;
  std::cout << "\nDone!\n";
  return NET;
}

std::vector<std::vector<int> > sample_distbase_trunc(
  std::vector<std::vector<double> > &net,
  std::vector<double> &bins, int &nbins,
  int &nodes, int &ncols, int &leaves,
  double &rho, double &lb, double &xmin, double &xmax
) {
  // Change random seed ----
  srand(time(0));
  // Declare variables ----
  int c, r, A, B, t = 0, sp = 5;
  double Rho = 0.0, d;
  // Declare network ----
  std::vector<std::vector<int> > NET(
    nodes, std::vector<int>(nodes, 0)
  );
  // Categorize distances ----
  std::vector<std::vector<std::vector<double> > > subnet;
  get_subnets(
    net, bins, leaves, nbins, subnet
  );
  std::cout << "Matching network density:\n";
  while (Rho < rho) {
    display_progress(Rho, rho, t, sp);
    // Generate random distance
    d =  (rand() % 1000) / 1000.;
    d = xmin - log(1. - d * (1. - exp(-lb * (xmax - xmin)))) / lb ;
    if (d > xmax) continue;
    c = find_bin(d, bins, nbins);
    if (c == -1) continue;
    if (subnet[c].size() == 0) continue;
    r = rand() % subnet[c].size();
    A = rand() % 2;
    if (A == 0) B = 1;
    else B = 0;
    NET[subnet[c][r][A]][subnet[c][r][B]]++;
    network_density(NET, ncols, Rho);
  }
  if (Rho > rho)
     NET[subnet[c][r][A]][subnet[c][r][B]]--;
  std::cout << "\nDone!\n";
  return NET;
}

std::vector<std::vector<int> > sample_distbase_M(
  std::vector<std::vector<double> > &net,
  std::vector<double> &bins, int &nbins,
  int &nrows, int &ncols, int &Drows,
  int &M, double &lb
) {
  // Change random seed ----
  srand(time(0));
  // Declare variables ----
  int c, r, A, B, m = 0, t = 0, sp = 5;
  double d;
  // Declare network ----
  std::vector<std::vector<int> > NET(
    nrows, std::vector<int>(nrows, 0)
  );
  // Categorize distances ----
  std::vector<std::vector<std::vector<double> > > subnet;
  get_subnets(
    net, bins, Drows, nbins, subnet
  );
  std::cout << "Matching network density:\n";
  while (m <= M) {
    display_progress(m, M, t, sp);
    // Generate random distance
    d = (rand() % 1000) / 1000.;
    d = - log(1 - d) / lb ;
    if (d > bins[nbins - 1]) continue;
    c = find_bin(d, bins, nbins);
    if (c == -1) continue;
    if (subnet[c].size() == 0) continue;
    r = rand() % subnet[c].size();
    A = rand() % 2;
    if (A == 0) B = 1;
    else B = 0;
    NET[subnet[c][r][A]][subnet[c][r][B]]++;
    network_M(NET, nrows, ncols, m);
  }
  std::cout << "\nDone!\n";
  return NET;
}

std::vector<std::vector<int> > const_sample_distbase_M(
  std::vector<std::vector<double> > &net,
  std::vector<double> &bins, int &nbins,
  int &nrows, int &ncols, int &Drows,
  int &M, int &number_neurons, double &lb
) {
  // Change random seed ----
  srand(time(0));
  // Declare variables ----
  int c, r, A, B, m = 0, t = 0, sp = 5, Count = 0;
  double d;
  // Declare network ----
  std::vector<std::vector<int> > NET(
    nrows, std::vector<int>(nrows, 0)
  );
  // Categorize distances ----
  std::vector<std::vector<std::vector<double> > > subnet;
  get_subnets(
    net, bins, Drows, nbins, subnet
  );
  std::cout << "Matching number of edges in injected areas:\n";
  while (m <= M) {
    display_progress(m, M, t, sp);
    d =  (rand() % 1000) / 1000.;
    d = - log(1 - d) / lb ;
    if (d > bins[nbins - 1]) continue;
    c = find_bin(d, bins, nbins);
    if (c == -1) continue;
    if (subnet[c].size() == 0) continue;
    r = rand() % subnet[c].size();
    A = rand() % 2;
    if (A == 0) B = 1;
    else B = 0;
    NET[subnet[c][r][A]][subnet[c][r][B]]++;
    network_M(NET, nrows, ncols, m);
  }
  // Filing up until reaching the same number of neurons ----
  std::cout << "\nMatching number of neurons:\n";
  t = 0;
  network_count_M(NET, nrows, ncols, Count);
  while (Count <= number_neurons) {
    display_progress(Count, number_neurons, t, sp);
    // Generate random distance
    d =  (rand() % 1000) / 1000.;
    d = - log(1 - d) / lb ;
    if (d > bins[nbins - 1]) continue;
    c = find_bin(d, bins, nbins);
    if (c == -1) continue;
    if (subnet[c].size() == 0) continue;
    r = rand() % subnet[c].size();
    A = rand() % 2;
    if (A == 0) B = 1;
    else B = 0;
    if (NET[subnet[c][r][A]][subnet[c][r][B]] == 0)
      continue;
    NET[subnet[c][r][A]][subnet[c][r][B]]++;
    if (subnet[c][r][A] < nrows && subnet[c][r][B] < ncols)
      Count++;
  }
  std::cout << "\nDone!\n";
  return NET;
}

std::vector<std::vector<int> > const_sample_distbase(
  std::vector<std::vector<double> > &net,
  std::vector<double> &bins, int &nbins,
  int &nodes, int &ncols, int &leaves, int &nlinks,
  double &rho, double &lb
) {
  // Change random seed ----
  srand(time(0));
  // Declare variables ----
  int c, r, A, B, t = 0, sp = 5, Count = 0;
  double Rho = 0.0, d;
  // Declare network ----
  std::vector<std::vector<int> > NET(
    nodes, std::vector<int>(nodes, 0)
  );
  // Categorize distances ----
  std::vector<std::vector<std::vector<double> > > subnet;
  get_subnets(
    net, bins, leaves, nbins, subnet
  );
  // Start getting right density ----
  std::cout << "Matching network density:\n";
  while (Rho < rho) {
    display_progress(Rho, rho, t, sp);
    d =  (rand() % 1000) / 1000.;
    d = - log(1 - d) / lb ;
    if (d > bins[nbins - 1]) continue;
    c = find_bin(d, bins, nbins);
    if (c == -1) continue;
    if (subnet[c].size() == 0) continue;
    r = rand() % subnet[c].size();
    A = rand() % 2;
    if (A == 0) B = 1;
    else B = 0;
    NET[subnet[c][r][A]][subnet[c][r][B]]++;
    network_density(NET, ncols, Rho);
  }
  if (Rho > rho)
     NET[subnet[c][r][A]][subnet[c][r][B]]--;
  std::cout << "\nDone!\n";
  // Start matching count number ----
  std::cout << "\nMatching count quantity:\n";
  t = 0;
  network_count(NET, nodes, Count);
  while (Count <= nlinks) {
    display_progress(Count, nlinks, t, sp);
    // Generate random distance
    d = (rand() % 1000) / 1000.;
    d = - log(1 - d) / lb ;
    if (d > bins[nbins - 1]) continue;
    c = find_bin(d, bins, nbins);
    if (c == -1) continue;
    if (subnet[c].size() == 0) continue;
    r = rand() % subnet[c].size();
    A = rand() % 2;
    if (A == 0) B = 1;
    else B = 0;
    if (NET[subnet[c][r][A]][subnet[c][r][B]] == 0)
      continue;
    NET[subnet[c][r][A]][subnet[c][r][B]]++;
    network_count_M(NET, nodes, ncols, Count);
  }
  std::cout << "\nDone!\n";
  return NET;
}

// swapnet ***

std::vector<std::vector<double> > swap_one_k(
  std::vector<std::vector<double> > &G,
  const int &rows, const int &cols, int &swaps
) {
  // Change random seed ----
  srand(time(0));
  // Declare the shuffle nodes ----
  int A, B, C, D, keep = 0, t = 0, sp = 5;
  int luck;
  double m;
  // Copy network ----
  std::vector<std::vector<double> > GG = G;
  // Start loop!!!
  printf("Swapping edges %i times:\n", swaps);
  while (t < swaps) {
    display_progress(t, swaps, keep, sp);
    A = rand() % rows;
    B = rand() % rows;
    C = rand() % cols;
    D = rand() % cols;
    if (A == B || C == D) continue;
    if (A == D || B == C) continue;
    if (A == C || B == D) continue;
    if (GG[A][C] == 0 || GG[B][D] == 0) continue;
    if (GG[A][D] > 0 || GG[B][C] > 0) continue;
    // Create luck ----
    luck = rand() % 2;
    switch (luck) {
    // Horizontal cases ----
    case 0:
      GG[A][D] = GG[A][C];
      GG[A][C] *= 0.0;
      GG[B][C] = GG[B][D];
      GG[B][D] *= 0.0;
      break;
    // Vertical case ----
    case 1:
      GG[B][C] = GG[A][C];
      GG[A][C] *= 0.0;
      GG[A][D] = GG[B][D];
      GG[B][D] *= 0.0;
      break;
    default:
      break;
    }
    t++;
  }
  std::cout << "\nDone!\n";
  return GG;
}

PYBIND11_MODULE(rand_network, m) {

  m.doc() = "Creates fast random networks";

  m.def(
    "sample_elliptic",
    &sample_elliptic,
    py::return_value_policy::reference_internal
  );

  m.def(
    "const_sample_elliptic",
    &const_sample_elliptic,
    py::return_value_policy::reference_internal
  );

  m.def(
    "sample_pareto",
    &sample_pareto,
    py::return_value_policy::reference_internal
  );

  m.def(
    "sample_pareto_trunc",
    &sample_pareto_trunc,
    py::return_value_policy::reference_internal
  );

  m.def(
    "const_sample_pareto",
    &const_sample_pareto,
    py::return_value_policy::reference_internal
  );

  m.def(
    "sample_distbase",
    &sample_distbase,
    py::return_value_policy::reference_internal
  );

  m.def(
    "sample_distbase_trunc",
    &sample_distbase_trunc,
    py::return_value_policy::reference_internal
  );

  m.def(
    "sample_distbase_M",
    &sample_distbase_M,
    py::return_value_policy::reference_internal
  );

  m.def(
    "const_sample_distbase_M",
    &const_sample_distbase_M,
    py::return_value_policy::reference_internal
  );

  m.def(
    "const_sample_distbase",
    &const_sample_distbase,
    py::return_value_policy::reference_internal
  );

  m.def(
    "swap_one_k",
    &swap_one_k,
    py::return_value_policy::reference_internal
  );

  m.def(
    "sample_from_prob",
    &sample_from_prob,
    py::return_value_policy::reference_internal
  );

  m.def(
    "const_sample_from_prob",
    &const_sample_from_prob,
    py::return_value_policy::reference_internal
  );
}