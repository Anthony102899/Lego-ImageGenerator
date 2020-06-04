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

Matrix<double, 3, 12> sharedLinearVelocity(Vector3d a1, Vector3d a2) {
    Matrix<double, 3, 12> mat;
    mat.row(0) << 1, 0, 0,      0,  a1[z], -a1[y], -1,  0,  0,      0, -a2[z],  a2[y];
    mat.row(1) << 0, 1, 0, -a1[z],      0,  a1[x],  0, -1,  0,  a2[z],      0, -a2[x];
    mat.row(2) << 0, 0, 1,  a1[y], -a1[x],      0,  0,  0, -1, -a2[y],  a2[x],      0;
    return mat;
}

MatrixXd constraintMatrixOfJoint(Vector3d a_screw, Vector3d a_nut, Vector3d u_screw, Vector3d u_nut) {
    MatrixXd mat = MatrixXd(5, 12);
    MatrixXd same_linear_velocity = sharedLinearVelocity(a_screw, a_nut);
    
    Vector3d u_ortho = u_nut.cross(u_screw).normalized();
    MatrixXd same_angular_velocity(2, 12);
    same_angular_velocity.row(0) << 
        0, 0, 0, u_nut[x], u_nut[y], u_nut[z], 0, 0, 0, -u_nut[x], -u_nut[y], -u_nut[z];
    same_angular_velocity.row(1) << 
        0, 0, 0, u_ortho[x], u_ortho[y], u_ortho[z], 0, 0, 0, -u_ortho[x], -u_ortho[y], -u_ortho[z];
    
    mat << same_linear_velocity, same_angular_velocity;
    return mat;
}

MatrixXd constraintMatrixOfPin(Vector3d a1, Vector3d a2, Vector3d u1, Vector3d u2) {
    MatrixXd mat = MatrixXd::Zero(5, 12);
                             
    Vector3d plane_normal = u1.cross(u2).normalized();
    // u1p perpenticular to u1 and the normal vector of the plane <u1, u2>
    Vector3d u1p = u1.cross(plane_normal).normalized();
    mat.block<3, 12>(0, 0) = sharedLinearVelocity(a1, a2);
    // constraints: same angular velocity along u1, and u1_p 
    mat.row(3) << 0, 0, 0, u1[x], u1[y], u1[z], 0, 0, 0, -u1[x], -u1[y], -u1[z];
    mat.row(4) << 0, 0, 0, u1p[x], u1p[y], u1p[z], 0, 0, 0, -u1p[x], -u1p[y], -u1p[z];

    return mat;
}

MatrixXd constraintMatrixOfAnchor(Vector3d a1, Vector3d a2, Vector3d u1, Vector3d u2) {
    // Vector3d u1 = a1.normalized();
    // Vector3d u2 = a2.normalized();
    Vector3d normal = u1.cross(u2).normalized();
    
    MatrixXd C_anchor(6, 12);
    MatrixXd C_pin = constraintMatrixOfPin(a1, a2, u1, u2);
    RowVectorXd c(12);
    c << 0, 0, 0,  normal[x],  normal[y],  normal[z], 
         0, 0, 0, -normal[x], -normal[y], -normal[z];
    C_anchor << C_pin, c;
    return C_anchor;
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
MatrixXd buildConstraintMatrix(MatrixXd P, MatrixXi E, MatrixXi pins, MatrixXi anchors) {

    MatrixXd mat = MatrixXd::Zero(pins.rows() * 5 + anchors.rows() * 6, E.rows() * 6);
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

        Vector3d a_vert = P.row(e_a(0));
        Vector3d b_vert = P.row(e_b(0));
        Vector3d u_a = a_a.isZero() ? (mid_a - a_vert).normalized() : a_a.normalized();
        Vector3d u_b = a_b.isZero() ? (mid_b - b_vert).normalized() : a_b.normalized();

        // if (i == 6 || i == 7 || i == 14 || i == 15) {
        //     oo("pin index", i);
        //     oo("e_a", e_a.transpose());
        //     oo("e_b", e_b.transpose());
        //     oo("pin vertex", vertex.transpose());
        //     oo("mid_a", mid_a.transpose());
        //     oo("edge_a_0", P.row(e_a(0)));
        //     oo("edge_a_1", P.row(e_a(1)));
        //     oo("mid_b", mid_b.transpose());
        //     oo("edge_b_0", P.row(e_b(0)));
        //     oo("edge_b_1", P.row(e_b(1)));
        //     oo("a_a", a_a.transpose());
        //     oo("a_b", a_b.transpose());
        // }
        if (u_a.cross(u_b).isZero()) {
            o("warning: u1 x u2 is zero vector, this will cause undefined behaviour");
            oo("u1xu2.normalized()", u_a.cross(u_b).transpose());
            oo(edge_a_index, edge_b_index);
        }
        // MatrixXd constraints = constraint_matrix_of_pin(a_a, a_b, u_a, u_b);
        MatrixXd constraints = constraintMatrixOfJoint(a_a, a_b, u_a, u_b);
        // o("copy blocks");
        mat.block<5, 6>(i * 5, edge_a_index * 6) = constraints.block<5, 6>(0, 0);
        mat.block<5, 6>(i * 5, edge_b_index * 6) = constraints.block<5, 6>(0, 6);
    }
    for (int i = 0; i < anchors.rows(); i++) {
        Vector3d vertex = P.row(anchors(i, 0));
        int edge_a_index = anchors(i, 1);
        int edge_b_index = anchors(i, 2);
        // o("get edges");
        Vector2i e_a = E.row(edge_a_index);
        Vector2i e_b = E.row(edge_b_index);
        Vector3d mid_a = (P.row(e_a(0)) + P.row(e_a(1))) / 2.0;
        Vector3d mid_b = (P.row(e_b(0)) + P.row(e_b(1))) / 2.0;
        Vector3d a_a = vertex - mid_a;
        Vector3d a_b = vertex - mid_b;

        Vector3d a_vert = P.row(e_a(0));
        Vector3d b_vert = P.row(e_b(0));
        Vector3d u_a = a_a.isZero() ? (mid_a - a_vert).normalized() : a_a.normalized();
        Vector3d u_b = a_b.isZero() ? (mid_b - b_vert).normalized() : a_b.normalized();

        if (u_a.cross(u_b).isZero()) {
            o("warning: u1 x u2 is zero vector, this will cause undefined behaviour");
            oo("u1xu2.normalized()", u_a.cross(u_b).transpose());
            oo(edge_a_index, edge_b_index);
        }
        MatrixXd constraints = constraintMatrixOfAnchor(a_a, a_b, u_a, u_b);
        int row_ind = i * 6 + pins.rows() * 5;
        mat.block<6, 6>(row_ind, edge_a_index * 6) = constraints.block<6, 6>(0, 0);
        mat.block<6, 6>(row_ind, edge_b_index * 6) = constraints.block<6, 6>(0, 6);
    }
    return mat;
}

void fixOneEdge(int index, VectorXd vw, MatrixXd &C, VectorXd &b) {
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

void fixOneVariable(int index, double value, MatrixXd C, VectorXd b, MatrixXd &newC, VectorXd &newb) {
    newC.resize(C.rows() + 1, C.cols());
    newb.resize(C.rows() + 1);
    RowVectorXd new_row = RowVectorXd::Zero(C.cols());
    new_row(index) = 1;
    newC << C, new_row;
    newb << b, value;
}

std::string getNameofIndex(int ind) {
    int e_ind = ind / 6;
    char vw = (ind % 6 <= 2) ? 'v' : 'w';
    int vw_ind = ind % 3;
    char xyz[] = "xyz";
    char buff[100];
    snprintf(buff, sizeof(buff), "e%d_%c%c", e_ind, vw, xyz[vw_ind]);
    std::string ret = buff;
    return ret;
}

bool solve(MatrixXd P, MatrixXi E, MatrixXi pins, MatrixXi anchors, int &dof, MatrixXd &constraints,
    std::vector<std::tuple<int, VectorXd, double>> &unstable_indices)
{
    MatrixXd C_init = buildConstraintMatrix(P, E, pins, anchors);
    VectorXd b = VectorXd::Zero(C_init.rows());
    VectorXd vw(6); vw << 0, 0, 0, 0, 0, 0;
    fixOneEdge(0, vw, C_init, b);
    auto C_dcmp = C_init.fullPivLu();
    auto compute_error = [](MatrixXd A, VectorXd x, VectorXd b) {
        return (A * x - b).norm();
    };

    int init_rank = C_dcmp.rank();
    dof = C_init.cols() - init_rank;
    bool full_rank = dof == 0;
    VectorXd sol = C_dcmp.solve(b);
    // double error = compute_error(C_init, sol, b);
    if (!full_rank) {
        for (int i = 0; i < C_init.cols(); i++) {
            MatrixXd C_i;
            VectorXd b_i;
            const int push_velocity = 10;
            fixOneVariable(i, push_velocity, C_init, b, C_i, b_i);
            auto C_decomp = C_i.fullPivHouseholderQr();
            VectorXd x = C_decomp.solve(b_i);
            int rank = C_decomp.rank();
            double err = compute_error(C_i, x, b_i);

            if (rank == init_rank + 1) {
                for (int m = 0; m < x.size(); m++) x(m) = (x(m) < 1e-11 && x(m) > -1e-11) ? 0 : x(m);
                unstable_indices.emplace_back(i, x, err);
            }
        }
    } else {
        oo("stable speed", sol.transpose());
        oo("stable error", compute_error(C_init, sol, b));
    }

    constraints = C_init;
    return unstable_indices.empty();
}