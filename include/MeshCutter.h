struct TupleHash {
    std::size_t operator()(const std::tuple<int, int, int>& t) const {
        std::size_t h1 = std::hash<int>()(std::get<0>(t));
        std::size_t h2 = std::hash<int>()(std::get<1>(t));
        std::size_t h3 = std::hash<int>()(std::get<2>(t));
        // Combine the hashes (simple example)
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

class MeshCutter {
public:
    MeshCutter(const Eigen::MatrixXd& V,
        const Eigen::MatrixXd& uv,
        const Eigen::MatrixXi& F,
        const Eigen::MatrixXi& FT);

    std::pair<Eigen::MatrixXd, Eigen::MatrixXi> cut_mesh();

    Eigen::MatrixXi load_feature_edges(std::string_view fe_filename);

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