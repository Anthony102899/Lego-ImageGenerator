#include <iostream>
#include <Eigen>

#include "reader.h"
#include "solver.h"

#define o(x) {std::cout << x << std::endl;}
#define oo(x, y) {std::cout << x << " " << y << std::endl;};

using namespace Eigen;

const int x = 0;
const int y = 1;
const int z = 2;

MatrixXd constraint_matrix_of_pin(Vector3d a1, Vector3d a2) {
    MatrixXd mat = MatrixXd::Zero(5, 12);

    Vector3d u1 = a1.normalized();
    Vector3d u2 = a2.normalized();
    Vector3d plane_normal = u1.cross(u2).normalized();
    // u1p perpenticular to u1 and the normal vector of the plane <u1, u2>
    Vector3d u1p = u1.cross(plane_normal).normalized();
    Vector3d u2p = u2.cross(plane_normal).normalized();

    mat.row(0) << 1, 0, 0,      0,  a1[z], -a1[y], -1,  0,  0,      0, -a2[z],  a2[y];
    mat.row(1) << 0, 1, 0, -a1[z],      0,  a1[x],  0, -1,  0,  a2[z],      0, -a2[x];
    mat.row(2) << 0, 0, 1,  a1[y], -a1[x],      0,  0,  0, -1, -a2[y],  a2[x],      0;

    // constraints: same angular velocity along u1, and u1_p 
    mat.row(3) << 0, 0, 0, u1[x], u1[y], u1[z], 0, 0, 0, -u1[x], -u1[y], -u1[z];
    mat.row(4) << 0, 0, 0, u1p[x], u1p[y], u1p[z], 0, 0, 0, -u1p[x], -u1p[y], -u1p[z];

    return mat;
}

/*
    Generate the constraints and store them into a matrix
    OUTPUT: Matrix of size (#pin*5 by #E*6), 
        rows are the constraints, row i to i + 4 correspond to the i-th pin
        cols are the linear and angular velocity of each edge, in the form of [v1 w1 v2 w2 ... vn wn]
    INPUT:
        P: (#P by 3) the vertices of the graph
        E: (#E by 2) the edges of the graph, store the indices into P of the vertices of each edge.
        pins: (#pin by (1 + 2) ) the pins in the shape, store
            the indices into P, of the vertex corresponding to the pin
            the indices into E, of the two edges that pin joins
*/
MatrixXd build_constraints_matrix(MatrixX3d P, MatrixX2i E, MatrixX3i pins) {
    assert(P.cols() == 3 && E.cols() == 2 && pins.cols() == 3);

    MatrixXd mat = MatrixXd::Zero(pins.rows() * 5, E.rows() * 6);
    for (int i = 0; i < pins.rows(); i++) {
        Vector3d vertex = P.row(pins(i, 0));
        int edge_a_index = pins(i, 1);
        int edge_b_index = pins(i, 2);
        // o("get edges");
        Vector2i e_a = E.row(edge_a_index);
        Vector2i e_b = E.row(edge_b_index);
        // o("get mid points");
        Vector3d mid_a = (P.row(e_a(0)) + P.row(e_a(1))) / 2.0;
        Vector3d mid_b = (P.row(e_b(0)) + P.row(e_b(1))) / 2.0;
        // o("get a");
        Vector3d a_a = vertex - mid_a;
        Vector3d a_b = vertex - mid_b;
        // o("compute constraints");
        MatrixXd constraints = constraint_matrix_of_pin(a_a, a_b);
        // o("copy blocks");
        mat.block<5, 6>(i * 5, edge_a_index * 6) = constraints.block<5, 6>(0, 0);
        mat.block<5, 6>(i * 5, edge_b_index * 6) = constraints.block<5, 6>(0, 6);
    }
    return mat;
}

void fix_one_edge(int index, VectorXd vw, MatrixXd &C, VectorXd &b) {
    MatrixXd newC(C.rows() + 6, C.cols());
    VectorXd newb(C.rows() + 6);

    MatrixXd new_rows = MatrixXd::Zero(6, C.cols());
    VectorXd new_b_entries = VectorXd::Zero(6);
    for (int i = 0; i < 6; i++) {
        new_rows(i, index * 6 + i) = 1;
        new_b_entries(i) = vw(i);
    }
    
    newC << C, new_rows;
    newb << b, new_b_entries;
    C = newC;
    b = newb;
}

void fix_one_variable(int index, double value, MatrixXd C, VectorXd b, MatrixXd &newC, VectorXd &newb) {
    newC.resize(C.rows() + 1, C.cols());
    newb.resize(C.rows() + 1);
    RowVectorXd new_row = RowVectorXd::Zero(C.cols());
    new_row(index) = 1;
    newC << C, new_row;
    newb << b, value;
}

bool solve(MatrixXd P, MatrixXi E, MatrixXi pins, int &dof,
    std::vector<std::pair<int, VectorXd>> &unstable_indices)
{
    MatrixXd C = build_constraints_matrix(P, E, pins);
    VectorXd b = VectorXd::Zero(C.rows());
    VectorXd vw(6); vw << 0, 0, 0, 0, 0, 0;
    fix_one_edge(0, vw, C, b);
    auto C_dcmp = C.fullPivLu();
    auto compute_error = [](MatrixXd A, VectorXd x, VectorXd b) {
        return (A * x - b).norm() / (b.array() + 1e-8).matrix().norm();
    };
    auto get_name_of_index = [](int ind) {
        int e_ind = ind / 6;
        char vw = (ind % 6 <= 2) ? 'v' : 'w';
        int vw_ind = ind % 3;
        char xyz[] = "xyz";
        char buff[100];
        snprintf(buff, sizeof(buff), "e%d_%c%c", e_ind, vw, xyz[vw_ind]);
        std::string ret = buff;
        return ret;
    };

    int rank = C_dcmp.rank();
    dof = C.cols() - rank;
    VectorXd sol = C_dcmp.solve(b);
    double error = compute_error(C, sol, b);
    if (C.cols() > rank) {
        for (int i = 0; i < C.cols(); i++) {
            MatrixXd C_i;
            VectorXd b_i;
            fix_one_variable(i, 10, C, b, C_i, b_i);
            auto C_decomp = C_i.fullPivHouseholderQr();
            VectorXd x = C_decomp.solve(b_i);
            int rank = C_decomp.rank();
            double err = compute_error(C_i, x, b_i);
            if (rank == C.cols()) {
                unstable_indices.emplace_back(i, x);
            }
        }
    }

    return unstable_indices.empty();
}

// int main() {
//     MatrixXd P;
//     MatrixXi E;
//     MatrixXi pins;
//     std::vector<int> unstable_indices;

//     trapezoid_data(P, E, pins);
//     // trapezoid_data(P, E, pins);
//     // two_triangles_data(P, E, pins);
//     solve(P, E, pins, unstable_indices);
//     for (int i = 0; i < unstable_indices.size(); i++) {
//         o(unstable_indices.at(i));
//     }

// }