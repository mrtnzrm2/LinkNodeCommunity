#include "../Sources/benchm.cpp"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

class BN {
private:
vector<vector<double> > network;
vector<vector<int> > communities;
public:
  BN(

		int N=100, double k=20, double maxk=30,
    double mut=0.2, double t1=2, double t2=3,
    int on=-1, int om=-1, int nmin=-1, int nmax=-1, double ca=-1., bool fixed_range=false, int seed=12345
  ) {
		srand5(seed);
		Parameters p;
    // Set parameters ----
    p.num_nodes = N;
    p.average_k = k;
    p.max_degree = maxk;
    p.mixing_parameter = mut;
    p.tau = t1;
    p.tau2 = t2;
    if (on > 0)
      p.overlap_membership = om;
    if (om > 0)
      p.overlapping_nodes = on;
    if (nmin > 0)
      p.nmin = nmin;
    if (nmax > 0)
      p.nmax = nmax;
		if (ca > 0)
			p.clustering_coeff = ca;
		if (fixed_range)
			p.fixed_range = fixed_range;
		// std::cout << nmin << " |||||| " << nmax << std::endl;
    create_network(
      p.excess, p.defect, p.num_nodes, p.average_k, p.max_degree,
      p.tau, p.tau2, p.mixing_parameter, p.overlapping_nodes,
			p.overlap_membership, p.nmin, p.nmax, p.fixed_range, p.clustering_coeff
    );
  }
  ~BN(){};
  int create_network(
		bool excess, bool defect, int num_nodes, double  average_k, int  max_degree, double  tau, double  tau2, 
		double  mixing_parameter, int  overlapping_nodes, int  overlap_membership, int  nmin, int  nmax, bool  fixed_range, double ca
  );
  int print_network(
    deque<set<int> > & E, const deque<deque<int> > & member_list, const deque<deque<int> > & member_matrix, deque<int> & num_seq
  );

	vector<vector<double> > get_network() {
		return network;
	}
	vector<vector<int> > get_communities() {
		return communities;
	}
};

int BN::create_network(
   bool excess, bool defect, int num_nodes, double  average_k, int  max_degree, double  tau, double  tau2, 
	double  mixing_parameter, int  overlapping_nodes, int  overlap_membership, int  nmin, int  nmax, bool  fixed_range, double ca
) {	
	
	
	
	
	double dmin=solve_dmin(max_degree, average_k, -tau);
	if (dmin==-1)
		return -1;
	
	int min_degree=int(dmin);
	
	
	double media1=integer_average(max_degree, min_degree, tau);
	double media2=integer_average(max_degree, min_degree+1, tau);
	
	if (fabs(media1-average_k)>fabs(media2-average_k))
		min_degree++;
	
	
	
	
	
	
		
	// range for the community sizes
	if (!fixed_range) {
	
		nmax=max_degree;
		nmin=max(int(min_degree), 3);
		cout<<"-----------------------------------------------------------"<<endl;
		cout<<"community size range automatically set equal to ["<<nmin<<" , "<<nmax<<"]"<<endl;
		

	}
	
	
	//----------------------------------------------------------------------------------------------------
	
	
	deque <int> degree_seq ;		//  degree sequence of the nodes
	deque <double> cumulative;
	powerlaw(max_degree, min_degree, tau, cumulative);
	
	for (int i=0; i<num_nodes; i++) {
		
		int nn=lower_bound(cumulative.begin(), cumulative.end(), ran4())-cumulative.begin()+min_degree;
		degree_seq.push_back(nn);
	
	}
	
	
	
	sort(degree_seq.begin(), degree_seq.end());
		
	if(deque_int_sum(degree_seq) % 2!=0)
		degree_seq[max_element(degree_seq.begin(), degree_seq.end()) - degree_seq.begin()]--;
	
	
	deque<deque<int> >  member_matrix;
	deque<int> num_seq;
	deque<int> internal_degree_seq;
	
	// ********************************			internal_degree and membership			***************************************************

	
	

	if(internal_degree_and_membership(mixing_parameter, overlapping_nodes, overlap_membership, num_nodes, member_matrix, excess, defect, degree_seq, num_seq, internal_degree_seq, fixed_range, nmin, nmax, tau2)==-1)
		return -1;
	
	
	
	deque<set<int> > E;					// E is the adjacency matrix written in form of list of edges
	deque<deque<int> > member_list;		// row i cointains the memberships of node i
	deque<deque<int> > link_list;		// row i cointains degree of the node i respect to member_list[i][j]; there is one more number that is the external degree

	
	
	std::cout<<"building communities... "<<std::endl;
	if(build_subgraphs(E, member_matrix, member_list, link_list, internal_degree_seq, degree_seq, excess, defect)==-1)
		return -1;	
	




	std::cout<<"connecting communities... "<<std::endl;
	connect_all_the_parts(E, member_list, link_list);
	


	if(erase_links(E, member_list, excess, defect, mixing_parameter)==-1)
		return -1;

	
	
	
	if(ca!=unlikely) {
		cout<<"trying to approach an average clustering coefficient ... "<<ca<<endl;
		cclu(E, member_list, member_matrix, ca);
	}
	
	
	
	
	cout<<"recording network..."<<endl;	
	print_network(E, member_list, member_matrix, num_seq);

	
	
	
		
	return 0;
}

int BN::print_network(deque<set<int> > & E, const deque<deque<int> > & member_list, const deque<deque<int> > & member_matrix, deque<int> & num_seq) {

	
	int edges=0;

		
	int num_nodes=member_list.size();
	
	deque<double> double_mixing;
	for (int i=0; i<E.size(); i++) {
		
		double one_minus_mu = double(internal_kin(E, member_list, i))/E[i].size();
		
		double_mixing.push_back(1.- one_minus_mu);
				
		edges+=E[i].size();
		
	}
	
	
	//cout<<"\n----------------------------------------------------------"<<endl;
	//cout<<endl;
	
	
	
	double density=0; 
	double sparsity=0;
	
	for (int i=0; i<member_matrix.size(); i++) {

		double media_int=0;
		double media_est=0;
		
		for (int j=0; j<member_matrix[i].size(); j++) {
			
			
			double kinj = double(internal_kin_only_one(E[member_matrix[i][j]], member_matrix[i]));
			media_int+= kinj;
			media_est+=E[member_matrix[i][j]].size() - double(internal_kin_only_one(E[member_matrix[i][j]], member_matrix[i]));
					
		}
		
		double pair_num=(member_matrix[i].size()*(member_matrix[i].size()-1));
		double pair_num_e=((num_nodes-member_matrix[i].size())*(member_matrix[i].size()));
		
		if(pair_num!=0)
			density+=media_int/pair_num;
		if(pair_num_e!=0)
			sparsity+=media_est/pair_num_e;
		
		
	
	}
	
	density=density/member_matrix.size();
	sparsity=sparsity/member_matrix.size();
	
	
	
	vector<double> net_tmp;

	// ofstream out1("network.dat");
	for (int u=0; u<E.size(); u++) {
	
		for(set<int>::iterator itb=E[u].begin(); itb!=E[u].end(); itb++) {
			net_tmp.push_back(u + 1);
			net_tmp.push_back(*(itb)+1);
			net_tmp.push_back(1);
			network.push_back(net_tmp);
			net_tmp.clear();
		}
			

	}
		
	vector<int> com_tmp;

	// ofstream out2("community.dat");
	for (int i=0; i<member_list.size(); i++) {
		com_tmp.push_back(i + 1);
		// out2<<i+1<<"\t";
		for (int j=0; j<member_list[i].size(); j++)
			com_tmp.push_back(member_list[i][j]+1);
			// out2<<member_list[i][j]+1<<" ";
		// out2<<endl;
		communities.push_back(com_tmp);
		com_tmp.clear();
	}
	// for (int i=0; i<member_list.size(); i++) {
		
	// 	std::cout<<i+1<<"\t";
	// 	for (int j=0; j<member_list[i].size(); j++)
	// 		std::cout<<member_list[i][j]+1<<" ";
	// 	std::cout<<std::endl;
	
	// }

	std::cout<<"\n\n---------------------------------------------------------------------------"<<std::endl;
	
	
	std::cout<<"network of "<<num_nodes<<" vertices and "<<edges/2<<" edges"<<";\t average degree = "<<double(edges)/num_nodes<<std::endl;
	std::cout<<"\naverage mixing parameter: "<<average_func(double_mixing)<<" +/- "<<sqrt(variance_func(double_mixing))<<std::endl;
	std::cout<<"p_in: "<<density<<"\tp_out: "<<sparsity<<std::endl;

	
	
	// ofstream statout("statistics.dat");
	
	// deque<int> degree_seq;
	// for (int i=0; i<E.size(); i++)
	// 	degree_seq.push_back(E[i].size());
	
	// std::cout<<"degree distribution (probability density function of the degree in logarithmic bins) "<<std::endl;
	// log_histogram(degree_seq, 10);
	// std::cout<<"\ndegree distribution (degree-occurrences) "<<std::endl;
	// int_histogram(degree_seq);
	// std::cout<<std::endl<<"--------------------------------------"<<std::endl;

		
	// std::cout<<"community distribution (size-occurrences)"<<std::endl;
	// int_histogram(num_seq);
	// std::cout<<std::endl<<"--------------------------------------"<<std::endl;

	// std::cout<<"mixing parameter"<<std::endl;
	// not_norm_histogram(double_mixing, 20, 0, 0);
	// std::cout<<std::endl<<"--------------------------------------"<<std::endl;
	
	
	


	std::cout<<std::endl<<std::endl;

	return 0;

}


PYBIND11_MODULE(BN, m) {
    py::class_<BN>(m, "BN")
        .def(
          py::init<int, double, double, double, double, double, int, int, int, int, double, bool, int>(),
					py::arg("N")=100, py::arg("k")=20, py::arg("maxk")=30,
					py::arg("mut")=0.2, py::arg("t1")=2, py::arg("t2")=3,
					py::arg("on")=-1, py::arg("om")=-1,
					py::arg("nmin")=-1, py::arg("nmax")=-1, py::arg("ca")=-1., py::arg("fixed_range")=false, py::arg("seed")=12345
        )
        .def("create_network", &BN::create_network)
        .def("print_network", &BN::print_network)
        .def("get_communities", &BN::get_communities)
				.def("get_network", &BN::get_network);
}