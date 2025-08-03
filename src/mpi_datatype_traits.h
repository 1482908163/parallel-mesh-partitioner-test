#ifndef MPI_DATATYPE_TRAITS_H
#define MPI_DATATYPE_TRAITS_H 

#include <mpi.h>
#include <cstdint>

namespace pmp {

template<typename T>
struct mpi_datatype_traits;

template<typename T>
constexpr MPI_Datatype mpi_datatype_v = mpi_datatype_traits<T>::value;

template<>
struct mpi_datatype_traits<int>
{
  static constexpr MPI_Datatype value = MPI_INT;
};

template<>
struct mpi_datatype_traits<std::int64_t>
{
  static constexpr MPI_Datatype value = MPI_INT64_T;
};

template<>
struct mpi_datatype_traits<float>
{
  static constexpr MPI_Datatype value = MPI_FLOAT;
};

template<>
struct mpi_datatype_traits<double>
{
  static constexpr MPI_Datatype value = MPI_DOUBLE;
};

// more types here ...

} // namespace pmp

#endif
