#include "../Sources/dir_benchm.cpp"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

class WDN {
private:
vector<vector<double> > network;
vector<vector<int> > communities;
public:
  WDN(
    int N=100, double k=20, double maxk=30,
    double mut=0.2, double muw=0.2,
    double beta=2, double t1=2, double t2=3,
    int on=-1, int om=-1, int nmin=-1, int nmax=-1
  ) {
		Parameters p;
    // Set parameters ----
    p.num_nodes = N;
    p.average_k = k;
    p.max_degree = maxk;
    p.mixing_parameter = mut;
    p.mixing_parameter2 = muw;
    p.beta = beta;
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
    create_network(
      p.excess, p.defect, p.num_nodes, p.average_k, p.max_degree,
      p.tau, p.tau2, p.mixing_parameter,  p.mixing_parameter2,
      p.beta, p.overlapping_nodes, p.overlap_membership, p.nmin, p.nmax, p.fixed_range
    );	
  }
  ~WDN(){};
  int create_network(
    bool excess, bool defect,
    int num_nodes, double  average_k, int  max_degree,
    double  tau, double  tau2, 
    double  mixing_parameter, double  mixing_parameter2,
    double  beta, int  overlapping_nodes, int  overlap_membership,
    int  nmin, int  nmax, bool  fixed_range
  );
  int weights(
    deque<set<int> > & ein, deque<set<int> > & eout,
    const deque<deque<int> > & member_list, const double beta, 
    const double mu,
    deque<map <int, double > > & neigh_weigh_in, deque<map <int, double > > & neigh_weigh_out
  );
  int print_network(
    deque<set<int> > & Ein, deque<set<int> > & Eout,
    const deque<deque<int> > & member_list, const deque<deque<int> > & member_matrix, 
	  deque<int> & num_seq,
    deque<map <int, double > > & neigh_weigh_in, deque<map <int, double > > & neigh_weigh_out,
    double beta, double mu, double mu0
  );
  int propagate(
    deque<map <int, double > > & neigh_weigh_in, deque<map <int, double > > & neigh_weigh_out,
    const deque<int> & VE, const deque<deque<int> > & member_list, 
	  deque<deque<double> > & wished, deque<deque<double> > & factual,
    int i, double & tot_var, double *strs, const deque<int> & internal_kin_top
  );

  int propagate_two(
    deque<map <int, double > > & neighbors_weights, const deque<int> & VE, const deque<deque<int> > & member_list, 
    deque<deque<double> > & wished, deque<deque<double> > & factual,
    int i, double & tot_var, double *strs, const deque<int> & internal_kin_top, deque<map <int, double > > & others
  );

  int propagate_one(
    deque<map <int, double > > & neighbors_weights, const deque<int> & VE,
    const deque<deque<int> > & member_list, 
    deque<deque<double> > & wished, deque<deque<double> > & factual,
    int i, double & tot_var, double *strs, const deque<int> & internal_kin_top, deque<map <int, double > > & others
  );
	vector<vector<double> > get_network() {
		return network;
	}
	vector<vector<int> > get_communities() {
		return communities;
	}
};

int WDN::create_network(
   bool excess, bool defect,
  int num_nodes, double  average_k, int  max_degree,
  double  tau, double  tau2, 
	double  mixing_parameter, double  mixing_parameter2,
  double  beta, int  overlapping_nodes, int  overlap_membership,
  int  nmin, int  nmax, bool  fixed_range
) {	
	// it finds the minimum degree ----
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

  // Something happening ----
	
	deque <int> degree_seq_in;		//  degree sequence of the nodes (in-links)
	deque <int> degree_seq_out;		//  degree sequence of the nodes (out-links)
	deque <double> cumulative;
	powerlaw(max_degree, min_degree, tau, cumulative);
	
	for (int i=0; i<num_nodes; i++) {
		
		int nn=lower_bound(cumulative.begin(), cumulative.end(), ran4())-cumulative.begin()+min_degree;
		degree_seq_in.push_back(nn);
	
	}
	
	sort(degree_seq_in.begin(), degree_seq_in.end());
		
	int inarcs=deque_int_sum(degree_seq_in);
	compute_internal_degree_per_node(inarcs, degree_seq_in.size(), degree_seq_out);
	
	deque<deque<int> >  member_matrix;
	deque<int> num_seq;
	deque<int> internal_degree_seq_in;
	deque<int> internal_degree_seq_out;
	
	// Internal_degree and membership ----

	if (internal_degree_and_membership(
    mixing_parameter, overlapping_nodes, overlap_membership,
    num_nodes, member_matrix, excess, defect,
    degree_seq_in, degree_seq_out, num_seq,
    internal_degree_seq_in, internal_degree_seq_out,
    fixed_range, nmin, nmax, tau2)==-1
  )
		return -1;
	
	deque<set<int> > Ein;				// Ein is the adjacency matrix written in form of list of edges (in-links)
	deque<set<int> > Eout;				// Eout is the adjacency matrix written in form of list of edges (out-links)
	deque<deque<int> > member_list;		// row i cointains the memberships of node i
	deque<deque<int> > link_list_in;	// row i cointains degree of the node i respect to member_list[i][j]; there is one more number that is the external degree (in-links)
	deque<deque<int> > link_list_out;	// row i cointains degree of the node i respect to member_list[i][j]; there is one more number that is the external degree (out-links)

	cout<<"building communities... "<<endl;
	if(build_subgraphs(Ein, Eout, member_matrix, member_list, link_list_in, link_list_out, internal_degree_seq_in, degree_seq_in, internal_degree_seq_out, degree_seq_out, excess, defect)==-1)
		return -1;	

	cout<<"connecting communities... "<<endl;
	connect_all_the_parts(Ein, Eout, member_list, link_list_in, link_list_out);
	
	if(erase_links(Ein, Eout, member_list, excess, defect, mixing_parameter)==-1)
		return -1;

  // Something happening ----
	
	deque<map <int, double > > neigh_weigh_in;
	deque<map <int, double > > neigh_weigh_out;

	cout<<"inserting weights..."<<endl;
	weights(Ein, Eout, member_list, beta, mixing_parameter2, neigh_weigh_in, neigh_weigh_out);

	cout<<"recording network..."<<endl;	
	print_network(Ein, Eout, member_list, member_matrix, num_seq, neigh_weigh_in, neigh_weigh_out, beta, mixing_parameter2, mixing_parameter);
	
	return 0;
}

int WDN::weights(
    deque<set<int> > & ein, deque<set<int> > & eout,
    const deque<deque<int> > & member_list, const double beta, 
    const double mu,
    deque<map <int, double > > & neigh_weigh_in, deque<map <int, double > > & neigh_weigh_out
  ) {

	double tstrength=0;
	deque<int> VE;							//VE is the degree of the nodes (in + out)
	deque<int> internal_kin_top;			//this is the internal degree of the nodes (in + out)

	for(int i=0; i<ein.size(); i++) {
		internal_kin_top.push_back(
      internal_kin(ein, member_list, i) + internal_kin(eout, member_list, i)
    );
		VE.push_back(ein[i].size()+eout[i].size());
		tstrength+=pow(VE[i], beta);
	}
	
	double strs[VE.size()]; // strength of the nodes
	// build a matrix like this: deque < map <int, double > > each row corresponds to link - weights 

	for(int i=0; i<VE.size(); i++) {
		map<int, double> new_map;
		neigh_weigh_in.push_back(new_map);
		neigh_weigh_out.push_back(new_map);
		for (set<int>::iterator its=ein[i].begin(); its!=ein[i].end(); its++)
			neigh_weigh_in[i].insert(make_pair(*its, 0.));
		for (set<int>::iterator its=eout[i].begin(); its!=eout[i].end(); its++)
			neigh_weigh_out[i].insert(make_pair(*its, 0.));
		strs[i]=pow(double(VE[i]), beta);
		//cout<<VE[i]<<" "<<strs[i]<<endl;
	}
  // Something happening ----
	deque<double> s_in_out_id_row(3);
	s_in_out_id_row[0]=0;
	s_in_out_id_row[1]=0;
	s_in_out_id_row[2]=0;
	
	deque<deque<double> > wished;	// 3 numbers for each node: internal, idle and extra strength. the sum of the three is strs[i]. wished is the theoretical, factual the factual one.
	deque<deque<double> > factual;
	
	for (int i=0; i<VE.size(); i++) {
		
		wished.push_back(s_in_out_id_row);
		factual.push_back(s_in_out_id_row);
		
	}
	double tot_var=0;
	for (int i=0; i<VE.size(); i++) {
		wished[i][0]=(1. -mu)*strs[i];
		wished[i][1]=mu*strs[i];
		factual[i][2]=strs[i];
		tot_var+= wished[i][0] * wished[i][0] + wished[i][1] * wished[i][1] + strs[i] * strs[i];
	}
	
	double precision = 1e-9;
	double precision2 = 1e-2;
	double not_better_than = pow(tstrength, 2) * precision;
	//cout<<"tot_var "<<tot_var<<";\tnot better "<<not_better_than<<endl;
	
	int step=0;
	while (true) {
		time_t t0=time(NULL);
		double pre_var=tot_var;
		for (int i=0; i<VE.size(); i++)
			propagate(neigh_weigh_in, neigh_weigh_out, VE, member_list, wished, factual, i, tot_var, strs, internal_kin_top);
		
		//check_weights(neigh_weigh_in, neigh_weigh_out, member_list, wished, factual, tot_var, strs);
		double relative_improvement=double(pre_var - tot_var)/pre_var;		
		//cout<<"tot_var "<<tot_var<<"\trelative improvement: "<<relative_improvement<<endl;
		if (tot_var<not_better_than)
				break;
		
		if (relative_improvement < precision2)
			break;	
		time_t t1= time(NULL);
		int deltat= t1 - t0;
		
		/*
		if(step%2==0 && deltat !=0)		
			cout<<"About "<<cast_int((log(not_better_than) -log(tot_var)) / log(1. - relative_improvement)) * deltat<<" secs..."<<endl;
		*/
		step++;
	}
	
	//check_weights(neigh_weigh_in, neigh_weigh_out, member_list, wished, factual, tot_var, strs);
	return 0;
}

int WDN::print_network(
  deque<set<int> > & Ein, deque<set<int> > & Eout,
  const deque<deque<int> > & member_list, const deque<deque<int> > & member_matrix, 
  deque<int> & num_seq,
  deque<map <int, double > > & neigh_weigh_in, deque<map <int, double > > & neigh_weigh_out,
  double beta, double mu, double mu0
) {

	int edges=0;
	int num_nodes=member_list.size();
	deque<double> double_mixing_in;
	for (int i=0; i<Ein.size(); i++) if(Ein[i].size()!=0) {
		double one_minus_mu = double(internal_kin(Ein, member_list, i))/Ein[i].size();
		double_mixing_in.push_back(fabs(1.- one_minus_mu));
		edges+=Ein[i].size();
	}
	deque<double> double_mixing_out;
	for (int i=0; i<Eout.size(); i++) if(Eout[i].size()!=0) {
		double one_minus_mu = double(internal_kin(Eout, member_list, i))/Eout[i].size();		
		double_mixing_out.push_back(fabs(1.- one_minus_mu));
	}	
	// Something happening ----
	double density=0; 
	double sparsity=0;
	for (int i=0; i<member_matrix.size(); i++) {
		double media_int=0;
		double media_est=0;
		for (int j=0; j<member_matrix[i].size(); j++) {			
			double kinj = double(internal_kin_only_one(Ein[member_matrix[i][j]], member_matrix[i]));
			media_int+= kinj;
			media_est+=Ein[member_matrix[i][j]].size() - double(internal_kin_only_one(Ein[member_matrix[i][j]], member_matrix[i]));
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
	for (int u=0; u<Eout.size(); u++) {
		for(set<int>::iterator itb=Eout[u].begin(); itb!=Eout[u].end(); itb++){
			net_tmp.push_back(u + 1);
			net_tmp.push_back(*(itb)+1);
			net_tmp.push_back(neigh_weigh_out[u][*(itb)]);
			network.push_back(net_tmp);
			net_tmp.clear();
		}
	}
	vector<int> com_tmp;
	for (int i=0; i<member_list.size(); i++) {
		com_tmp.push_back(i + 1);
		for (int j=0; j<member_list[i].size(); j++)
			com_tmp.push_back(member_list[i][j]+1);
		communities.push_back(com_tmp);
		com_tmp.clear();
	}
  // Printing -----
	cout<<"\n\n---------------------------------------------------------------------------"<<endl;
	cout<<"network of "<<num_nodes<<" vertices and "<<edges<<" edges"<<";\t average degree = "<<double(edges)/num_nodes<<endl;
	cout<<"\naverage mixing parameter (in-links): "<<average_func(double_mixing_in)<<" +/- "<<sqrt(variance_func(double_mixing_in))<<endl;
	cout<<"average mixing parameter (out-links): "<<average_func(double_mixing_out)<<" +/- "<<sqrt(variance_func(double_mixing_out))<<endl;
	cout<<"p_in: "<<density<<"\tp_out: "<<sparsity<<endl;
	return 0;
}

int WDN::propagate_one(
  deque<map <int, double > > & neighbors_weights, const deque<int> & VE,
  const deque<deque<int> > & member_list, 
	deque<deque<double> > & wished, deque<deque<double> > & factual,
  int i, double & tot_var, double *strs, const deque<int> & internal_kin_top, deque<map <int, double > > & others
) {		
	double change=factual[i][2]/VE[i];
	double oldpartvar=0;
	for(map<int, double>::iterator itm=neighbors_weights[i].begin(); itm!=neighbors_weights[i].end(); itm++) if(itm->second + change > 0)
		for (int bw=0; bw<3; bw++) 
			oldpartvar+= (factual[itm->first][bw] - wished[itm->first][bw]) * (factual[itm->first][bw] - wished[itm->first][bw]);
	for (int bw=0; bw<3; bw++)
		oldpartvar+= (factual[i][bw] - wished[i][bw]) * (factual[i][bw] - wished[i][bw]);
	double newpartvar=0;
	for(
    map<int, double>::iterator itm=neighbors_weights[i].begin(); itm!=neighbors_weights[i].end(); itm++
  ) if(itm->second + change > 0) {
		if (they_are_mate(i, itm->first, member_list)) {
			factual[itm->first][0]+=change;
			factual[itm->first][2]-=change;
			factual[i][0]+=change;
			factual[i][2]-=change;
		}
		else {
			factual[itm->first][1]+=change;
			factual[itm->first][2]-=change;
			factual[i][1]+=change;
			factual[i][2]-=change;	
		}
		for (int bw=0; bw<3; bw++)
			newpartvar+= (factual[itm->first][bw] - wished[itm->first][bw]) * (factual[itm->first][bw] - wished[itm->first][bw]);
		itm->second+= change;
		others[itm->first][i]+=change;
	}
	for (int bw=0; bw<3; bw++)
		newpartvar+= (factual[i][bw] - wished[i][bw]) * (factual[i][bw] - wished[i][bw]);
	tot_var+= newpartvar - oldpartvar;
	return 0;
}


int WDN::propagate_two(
  deque<map <int, double > > & neighbors_weights, const deque<int> & VE, const deque<deque<int> > & member_list, 
  deque<deque<double> > & wished, deque<deque<double> > & factual,
  int i, double & tot_var, double *strs, const deque<int> & internal_kin_top, deque<map <int, double > > & others
) {

	int internal_neigh=internal_kin_top[i];
	if(internal_neigh!=0) {		// in this case I rewire the difference strength
		double changenn=(factual[i][0] - wished[i][0]);
		double oldpartvar=0;
		for(map<int, double>::iterator itm=neighbors_weights[i].begin(); itm!=neighbors_weights[i].end(); itm++) {
			if(they_are_mate(i, itm->first, member_list)) {
				double change = changenn/internal_neigh;
				if(itm->second - change > 0)
					for (int bw=0; bw<3; bw++) 
						oldpartvar+= (
              factual[itm->first][bw] - wished[itm->first][bw]
            ) * (
              factual[itm->first][bw] - wished[itm->first][bw]
            );				
			}
			else {
				double change = changenn/(VE[i] - internal_neigh);
				if(itm->second + change > 0)
					for (int bw=0; bw<3; bw++) 
						oldpartvar+= (
              factual[itm->first][bw] - wished[itm->first][bw]
            ) * (
              factual[itm->first][bw] - wished[itm->first][bw]
            );
			}
		}
		
		for (int bw=0; bw<3; bw++)
			oldpartvar+= (factual[i][bw] - wished[i][bw]) * (factual[i][bw] - wished[i][bw]);

		double newpartvar=0;
		for(map<int, double>::iterator itm=neighbors_weights[i].begin(); itm!=neighbors_weights[i].end(); itm++) {
			if (they_are_mate(i, itm->first, member_list)) {
				double change = changenn/internal_neigh;
				if(itm->second - change > 0) {
					factual[itm->first][0]-=change;
					factual[itm->first][2]+=change;
					factual[i][0]-=change;
					factual[i][2]+=change;
					for (int bw=0; bw<3; bw++)
						newpartvar+= (
              factual[itm->first][bw] - wished[itm->first][bw]
            ) * (factual[itm->first][bw] - wished[itm->first][bw]);

					itm->second-= change;
					others[itm->first][i]-=change;
				}		
			}
			else {
				double change = changenn/(VE[i] - internal_neigh);
				if(itm->second + change > 0) {
					factual[itm->first][1]+=change;
					factual[itm->first][2]-=change;
					
					factual[i][1]+=change;
					factual[i][2]-=change;
					for (int bw=0; bw<3; bw++)
						newpartvar+= (factual[itm->first][bw] - wished[itm->first][bw]) * (factual[itm->first][bw] - wished[itm->first][bw]);

					itm->second+= change;
					others[itm->first][i]+=change;	
				}		
			}
		}
	
		for (int bw=0; bw<3; bw++)
			newpartvar+= (factual[i][bw] - wished[i][bw]) * (factual[i][bw] - wished[i][bw]);		
		tot_var+=newpartvar - oldpartvar;
	}
	return 0;
}

int WDN::propagate(
  deque<map <int, double > > & neigh_weigh_in, deque<map <int, double > > & neigh_weigh_out,
  const deque<int> & VE, const deque<deque<int> > & member_list, 
  deque<deque<double> > & wished, deque<deque<double> > & factual,
  int i, double & tot_var, double *strs, const deque<int> & internal_kin_top
) {

	propagate_one(neigh_weigh_in, VE, member_list, wished, factual, i, tot_var, strs, internal_kin_top, neigh_weigh_out);
	propagate_one(neigh_weigh_out, VE, member_list, wished, factual, i, tot_var, strs, internal_kin_top, neigh_weigh_in);
	propagate_two(neigh_weigh_in, VE, member_list, wished, factual, i, tot_var, strs, internal_kin_top, neigh_weigh_out);
	propagate_two(neigh_weigh_out, VE, member_list, wished, factual, i, tot_var, strs, internal_kin_top, neigh_weigh_in);

	//check_weights(neigh_weigh_in, neigh_weigh_out, member_list, wished, factual, tot_var, strs);
	return 0;
}

PYBIND11_MODULE(WDN, m) {
    py::class_<WDN>(m, "WDN")
        .def(
          py::init<int, double, double, double, double, double, double, double, int, int, int, int>(),
					py::arg("N")=100, py::arg("k")=20, py::arg("maxk")=30,
					py::arg("mut")=0.2, py::arg("muw")=0.2,
					py::arg("beta")=2, py::arg("t1")=2, py::arg("t2")=3,
					py::arg("on")=-1, py::arg("om")=-1,
					py::arg("nmin")=-1, py::arg("nmax")=-1
        )
        .def("create_network", &WDN::create_network)
        .def("print_network", &WDN::print_network)
        .def("weights", &WDN::weights)
        .def("propagate", &WDN::propagate)
        .def("propagate_one", &WDN::propagate_one)
        .def("propagate_two", &WDN::propagate_two)
        .def("get_communities", &WDN::get_communities)
				.def("get_network", &WDN::get_network);
}