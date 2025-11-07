
#include <tuple>

class MeshCutter {
public:
    MeshCutter(const Eigen::MatrixXd& V,
        const Eigen::MatrixXd& uv,
        const Eigen::MatrixXi& F,
        const Eigen::MatrixXi& FT);

    std::pair<Eigen::MatrixXd, Eigen::MatrixXi> cut_mesh();

    Eigen::MatrixXi load_feature_edges(const std::string& fe_filename);

    Eigen::MatrixXi reindex_feature_edges(const Eigen::MatrixXi& FE);

    Eigen::MatrixXi remove_cycles_and_duplicates(const Eigen::MatrixXi& FE, const Eigen::MatrixXi& FE_reindex);

private:
    Eigen::MatrixXd V;
    Eigen::MatrixXd uv;
    Eigen::MatrixXi F;
    Eigen::MatrixXi FT;
    Eigen::MatrixXi TT;
    Eigen::MatrixXi TTi;
};