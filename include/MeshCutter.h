class MeshCutter {
public:
    MeshCutter(const Eigen::MatrixXd& V,
        const Eigen::MatrixXd& uv,
        const Eigen::MatrixXi& F,
        const Eigen::MatrixXi& FT);

    std::pair<Eigen::MatrixXd, Eigen::MatrixXi> cut_mesh();

    Eigen::MatrixXi load_feature_edges(std::string_view fe_filename);

    Eigen::MatrixXi reindex_feature_edges(const Eigen::MatrixXi& FE);

private:
    Eigen::MatrixXd V;
    Eigen::MatrixXd uv;
    Eigen::MatrixXi F;
    Eigen::MatrixXi FT;
    Eigen::MatrixXi TT;
    Eigen::MatrixXi TTi;
};