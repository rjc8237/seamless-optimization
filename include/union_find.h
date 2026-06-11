// Copyright (C) 2026 Ryan Capouellez
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

#pragma once
#include <vector>
#include <numeric>

namespace SymDir {

    /**
     * @brief Simple implementation of a union find data structure
     */
    class UnionFind {
    public:
        /**
         * @brief Construct a new Union Find object of a given (fixed) size
         *
         * @param size: number of elements in the set
         */
        UnionFind(int size) {
            // Each element is originally a root
            m_parent.resize(size);
            std::iota(m_parent.begin(), m_parent.end(), 0);
        }

        bool is_root(int index) const { return (m_parent[index] == index); }

        int find_set(int index) {
            // Get root index of set
            int root = index;
            while (!is_root(root)) {
                root = m_parent[root];
            }

            // Overwrite existing parents for faster traversal later
            int write_index = index;
            while (m_parent[write_index] != root) {
                int parent_index = m_parent[write_index];
                m_parent[write_index] = parent_index;
                write_index = parent_index;
            }

            // Return root
            return root;
        }

        void union_sets(int first_index, int second_index) {
            // Get set indices of two element indices
            int first_set = find_set(first_index);
            int second_set = find_set(second_index);

            // Do nothing if already same set
            if (first_set == second_set) return;

            // Union two sets arbitrarily
            // TODO Use better (size or rank) update rule
            m_parent[second_set] = first_set;
        }

        int count_elements() const { return m_parent.size(); }

        int count_sets() {
            // Count sets by counting roots
            int size = count_elements();
            int num_roots = 0;
            for (int i = 0; i < size; ++i) {
                if (is_root(i)) ++num_roots;
            }

            return num_roots;
        }

        std::vector<int> index_sets() {
            // Count sets by counting roots
            int num_elements = count_elements();

            // Make map from roots to sets
            int count = 0;
            std::vector<int> set_index(num_elements, -1);
            for (int i = 0; i < num_elements; ++i) {
                if (is_root(i)) {
                    set_index[i] = count;
                    count++;
                }
            }

            // Index remaining elements
            for (int i = 0; i < num_elements; ++i) {
                set_index[i] = set_index[find_set(i)];
            }

            return set_index;
        }

        std::vector<std::vector<int>> build_sets() {
            // Count sets by counting roots
            int num_elements = count_elements();
            int num_sets = count_sets();
            std::vector<int> index = index_sets();
            std::vector<std::vector<int>> sets(num_sets, std::vector<int>({}));
            for (int i = 0; i < num_elements; ++i) {
                sets[index[i]].push_back(i);
            }

            return sets;
        }

    private:
        std::vector<int> m_parent{};
    };
}
