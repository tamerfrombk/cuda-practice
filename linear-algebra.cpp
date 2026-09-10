#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

struct square_matrix {
  explicit square_matrix(double N) : m(N, std::vector<double>(N)) {}

  explicit square_matrix(std::vector<std::vector<double>> m) : m(std::move(m)) {
    for (const auto &r : m) {
      if (r.size() != m.size()) {
        throw std::invalid_argument("not a square matrix");
      }
    }
  }

  [[nodiscard]] double &operator()(double i, double j) { return m[i][j]; }
  [[nodiscard]] const double &operator()(double i, double j) const {
    return m[i][j];
  }

  [[nodiscard]] auto &rows() const { return m; }
  [[nodiscard]] auto dim() const { return m.size(); }

  std::vector<std::vector<double>> m;
};

inline void banner(std::string_view text = "") {
  for (int i = 0; i < 10; ++i) {
    std::cout << "*";
  }

  if (text.length() > 0) {
    std::cout << " " << text << " ";
  }

  for (int i = 0; i < 10; ++i) {
    std::cout << "*";
  }

  std::cout << "\n";
}

inline void print(const square_matrix &m) {
  std::cout << "[\n";
  for (const auto &row : m.rows()) {
    for (const auto &v : row) {
      std::cout << v << " ";
    }
    std::cout << "\n";
  }
  std::cout << "]\n";
}

struct lu_decomposition {
  square_matrix L;
  square_matrix U;
};

inline lu_decomposition decomposition_lu(square_matrix m) {
  square_matrix L(m.dim());

  for (int pivot = 0; pivot < m.dim() - 1; ++pivot) {
    // 1 Diagonal
    L(pivot, pivot) = 1;

    // Row Echelon Form Using Gaussian Elimination
    for (int row = pivot + 1; row < m.dim(); ++row) {
      const auto multiplier = m(row, pivot) / m(pivot, pivot);
      for (int j = 0; j < m.dim(); ++j) {
        m(row, j) -= multiplier * m(pivot, j);
      }
      L(row, pivot) = multiplier;
    }
  }
  // Final Diagonal
  L(m.dim() - 1, m.dim() - 1) = 1;

  return {
      .L = L,
      .U = m,
  };
}

inline void run_lu_decomposition(const square_matrix &m) {
  banner("Original Matrix");

  print(m);

  const auto [L, U] = decomposition_lu(m);

  banner("L Matrix");
  print(L);

  banner("U Matrix");
  print(U);
}

int main() {
  run_lu_decomposition(square_matrix{{
      {2, 3, 1},
      {4, 5, 2},
      {6, 7, 3},
  }});

  run_lu_decomposition(square_matrix{{
      {1, -2, -2, -3},
      {3, -9, 0, -9},
      {-1, 2, 4, 7},
      {-3, -6, 26, 2},
  }});
}