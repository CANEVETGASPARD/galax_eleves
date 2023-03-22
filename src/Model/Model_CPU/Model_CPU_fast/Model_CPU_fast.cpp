#include <cmath>
#include <iostream>

#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>

namespace xs = xsimd;

Model_CPU_fast ::Model_CPU_fast(const Initstate &initstate, Particles &particles)
    : Model_CPU(initstate, particles)
{
}

void Model_CPU_fast ::step()
{
    omp_set_num_threads(4);
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0);

#ifdef OMP
    // OMP  version
#pragma omp parallel for
    for (int i = 0; i < n_particles; i++)
    {
        for (int j = 0; j < n_particles; j++)
        {
            if (i != j)
            {
                const float diffx = particles.x[j] - particles.x[i];
                const float diffy = particles.y[j] - particles.y[i];
                const float diffz = particles.z[j] - particles.z[i];

                float dij = diffx * diffx + diffy * diffy + diffz * diffz;

                if (dij < 1.0)
                {
                    dij = 10.0;
                }
                else
                {
                    dij = std::sqrt(dij);
                    dij = 10.0 / (dij * dij * dij);
                }

                accelerationsx[i] += diffx * dij * initstate.masses[j];
                accelerationsy[i] += diffy * dij * initstate.masses[j];
                accelerationsz[i] += diffz * dij * initstate.masses[j];
            }
        }
    }

#pragma omp parallel for
    for (int i = 0; i < n_particles; i++)
    {
        velocitiesx[i] += accelerationsx[i] * 2.0f;
        velocitiesy[i] += accelerationsy[i] * 2.0f;
        velocitiesz[i] += accelerationsz[i] * 2.0f;
        particles.x[i] += velocitiesx[i] * 0.1f;
        particles.y[i] += velocitiesy[i] * 0.1f;
        particles.z[i] += velocitiesz[i] * 0.1f;
    }
#endif

#ifdef XSIMD
    // XSIMD version
    using b_type = xs::batch<float>;
    b_type b = 10.0;

    std::size_t inc = b_type::size;
    // size for which the vectorization is possible
    std::size_t vec_size = n_particles - n_particles % inc;
    for (int i = 0; i < vec_size; i += inc)
    {
        // load registers body i
        const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
        b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
        b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

        for (int j = 0; j < n_particles; j++)
        {
            if (j > i && j < i + inc - 1)
            {
                for (int k = i; k < i + inc; k++)
                {
                    if (k != j)
                    {
                        const float diffx = particles.x[j] - particles.x[i];
                        const float diffy = particles.y[j] - particles.y[i];
                        const float diffz = particles.z[j] - particles.z[i];

                        float dij = diffx * diffx + diffy * diffy + diffz * diffz;

                        if (dij < 1.0)
                        {
                            dij = 10.0;
                        }
                        else
                        {
                            dij = std::sqrt(dij);
                            dij = 10.0 / (dij * dij * dij);
                        }

                        accelerationsx[i] += diffx * dij * initstate.masses[j];
                        accelerationsy[i] += diffy * dij * initstate.masses[j];
                        accelerationsz[i] += diffz * dij * initstate.masses[j];

                        raccx_i = b_type::load_unaligned(&accelerationsx[i]);
                        raccy_i = b_type::load_unaligned(&accelerationsy[i]);
                        raccz_i = b_type::load_unaligned(&accelerationsz[i]);
                    }
                }
            }
            else
            {
                const b_type rposx_j = particles.x[j];
                const b_type rposy_j = particles.y[j];
                const b_type rposz_j = particles.z[j];
                const b_type diffx = rposx_j - rposx_i;
                const b_type diffy = rposy_j - rposy_i;
                const b_type diffz = rposz_j - rposz_i;

                b_type dij = diffx * diffx + diffy * diffy + diffz * diffz;
                dij = xs::sqrt(dij);
                dij = 10.0 / (dij * dij * dij);

                dij = xs::fmin(dij, b);

                raccx_i += diffx * dij * initstate.masses[j];
                raccy_i += diffy * dij * initstate.masses[j];
                raccz_i += diffz * dij * initstate.masses[j];

                raccx_i.store_unaligned(&accelerationsx[i]);
                raccy_i.store_unaligned(&accelerationsy[i]);
                raccz_i.store_unaligned(&accelerationsz[i]);
            }
        }
    }

    for (int i = vec_size; i < n_particles; i++)
    {
        for (int j = 0; j < n_particles; j++)
        {
            if (i != j)
            {
                const float diffx = particles.x[j] - particles.x[i];
                const float diffy = particles.y[j] - particles.y[i];
                const float diffz = particles.z[j] - particles.z[i];

                float dij = diffx * diffx + diffy * diffy + diffz * diffz;

                if (dij < 1.0)
                {
                    dij = 10.0;
                }
                else
                {
                    dij = std::sqrt(dij);
                    dij = 10.0 / (dij * dij * dij);
                }

                accelerationsx[i] += diffx * dij * initstate.masses[j];
                accelerationsy[i] += diffy * dij * initstate.masses[j];
                accelerationsz[i] += diffz * dij * initstate.masses[j];
            }
        }
    }

    //     for (int i = 0; i < n_particles; i++)
    //     {
    //         velocitiesx[i] += accelerationsx[i] * 2.0f;
    //         velocitiesy[i] += accelerationsy[i] * 2.0f;
    //         velocitiesz[i] += accelerationsz[i] * 2.0f;
    //         particles.x[i] += velocitiesx[i] * 0.1f;
    //         particles.y[i] += velocitiesy[i] * 0.1f;
    //         particles.z[i] += velocitiesz[i] * 0.1f;
    //     }

    for (int i = 0; i < vec_size; i += inc)
    {
        b_type rvelz_i = b_type::load_unaligned(&velocitiesz[i]);
        b_type rvelx_i = b_type::load_unaligned(&velocitiesx[i]);
        b_type rvely_i = b_type::load_unaligned(&velocitiesy[i]);
        const b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
        const b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        const b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);
        b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        b_type rposz_i = b_type::load_unaligned(&particles.z[i]);

        rvelx_i += raccx_i * 2.0f;
        rvely_i += raccy_i * 2.0f;
        rvelz_i += raccz_i * 2.0f;
        rposx_i += rvelx_i * 0.1f;
        rposy_i += rvely_i * 0.1f;
        rposz_i += rvelz_i * 0.1f;

        rvelx_i.store_unaligned(&velocitiesx[i]);
        rvely_i.store_unaligned(&velocitiesy[i]);
        rvelz_i.store_unaligned(&velocitiesz[i]);

        rposx_i.store_unaligned(&particles.x[i]);
        rposy_i.store_unaligned(&particles.y[i]);
        rposz_i.store_unaligned(&particles.z[i]);
    }

    for (int i = vec_size; i < n_particles; i++)
    {
        velocitiesx[i] += accelerationsx[i] * 2.0f;
        velocitiesy[i] += accelerationsy[i] * 2.0f;
        velocitiesz[i] += accelerationsz[i] * 2.0f;
        particles.x[i] += velocitiesx[i] * 0.1f;
        particles.y[i] += velocitiesy[i] * 0.1f;
        particles.z[i] += velocitiesz[i] * 0.1f;
    }

#endif
#ifndef XSIMD_OMP
    // OMP + xsimd version
    using b_type = xs::batch<float>;
    b_type b = 10.0;

    std::size_t inc = b_type::size;
    // size for which the vectorization is possible
    std::size_t vec_size = n_particles - n_particles % inc;
#pragma omp parallel for
    for (int i = 0; i < vec_size; i += inc)
    {
        // load registers body i
        const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
        b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
        b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

        for (int j = 0; j < n_particles; j++)
        {
            if (j > i && j < i + inc - 1)
            {
                for (int k = i; k < i + inc; k++)
                {
                    if (k != j)
                    {
                        const float diffx = particles.x[j] - particles.x[i];
                        const float diffy = particles.y[j] - particles.y[i];
                        const float diffz = particles.z[j] - particles.z[i];

                        float dij = diffx * diffx + diffy * diffy + diffz * diffz;

                        if (dij < 1.0)
                        {
                            dij = 10.0;
                        }
                        else
                        {
                            dij = std::sqrt(dij);
                            dij = 10.0 / (dij * dij * dij);
                        }

                        accelerationsx[i] += diffx * dij * initstate.masses[j];
                        accelerationsy[i] += diffy * dij * initstate.masses[j];
                        accelerationsz[i] += diffz * dij * initstate.masses[j];

                        raccx_i = b_type::load_unaligned(&accelerationsx[i]);
                        raccy_i = b_type::load_unaligned(&accelerationsy[i]);
                        raccz_i = b_type::load_unaligned(&accelerationsz[i]);
                    }
                }
            }
            else
            {
                const b_type rposx_j = particles.x[j];
                const b_type rposy_j = particles.y[j];
                const b_type rposz_j = particles.z[j];
                const b_type diffx = rposx_j - rposx_i;
                const b_type diffy = rposy_j - rposy_i;
                const b_type diffz = rposz_j - rposz_i;

                b_type dij = diffx * diffx + diffy * diffy + diffz * diffz;
                dij = xs::sqrt(dij);
                dij = 10.0 / (dij * dij * dij);

                dij = xs::fmin(dij, b);

                raccx_i += diffx * dij * initstate.masses[j];
                raccy_i += diffy * dij * initstate.masses[j];
                raccz_i += diffz * dij * initstate.masses[j];

                raccx_i.store_unaligned(&accelerationsx[i]);
                raccy_i.store_unaligned(&accelerationsy[i]);
                raccz_i.store_unaligned(&accelerationsz[i]);
            }
        }
    }

#pragma omp parallel for
    for (int i = vec_size; i < n_particles; i++)
    {
        for (int j = 0; j < n_particles; j++)
        {
            if (i != j)
            {
                const float diffx = particles.x[j] - particles.x[i];
                const float diffy = particles.y[j] - particles.y[i];
                const float diffz = particles.z[j] - particles.z[i];

                float dij = diffx * diffx + diffy * diffy + diffz * diffz;

                if (dij < 1.0)
                {
                    dij = 10.0;
                }
                else
                {
                    dij = std::sqrt(dij);
                    dij = 10.0 / (dij * dij * dij);
                }

                accelerationsx[i] += diffx * dij * initstate.masses[j];
                accelerationsy[i] += diffy * dij * initstate.masses[j];
                accelerationsz[i] += diffz * dij * initstate.masses[j];
            }
        }
    }

    // #pragma omp parallel for
    //     for (int i = 0; i < n_particles; i++)
    //     {
    //         velocitiesx[i] += accelerationsx[i] * 2.0f;
    //         velocitiesy[i] += accelerationsy[i] * 2.0f;
    //         velocitiesz[i] += accelerationsz[i] * 2.0f;
    //         particles.x[i] += velocitiesx[i] * 0.1f;
    //         particles.y[i] += velocitiesy[i] * 0.1f;
    //         particles.z[i] += velocitiesz[i] * 0.1f;
    //     }

#pragma omp parallel for
    for (int i = 0; i < vec_size; i += inc)
    {
        b_type rvelz_i = b_type::load_unaligned(&velocitiesz[i]);
        b_type rvelx_i = b_type::load_unaligned(&velocitiesx[i]);
        b_type rvely_i = b_type::load_unaligned(&velocitiesy[i]);
        const b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
        const b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        const b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);
        b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        b_type rposz_i = b_type::load_unaligned(&particles.z[i]);

        rvelx_i += raccx_i * 2.0f;
        rvely_i += raccy_i * 2.0f;
        rvelz_i += raccz_i * 2.0f;
        rposx_i += rvelx_i * 0.1f;
        rposy_i += rvely_i * 0.1f;
        rposz_i += rvelz_i * 0.1f;

        rvelx_i.store_unaligned(&velocitiesx[i]);
        rvely_i.store_unaligned(&velocitiesy[i]);
        rvelz_i.store_unaligned(&velocitiesz[i]);

        rposx_i.store_unaligned(&particles.x[i]);
        rposy_i.store_unaligned(&particles.y[i]);
        rposz_i.store_unaligned(&particles.z[i]);
    }

#pragma omp parallel for
    for (int i = vec_size; i < n_particles; i++)
    {
        velocitiesx[i] += accelerationsx[i] * 2.0f;
        velocitiesy[i] += accelerationsy[i] * 2.0f;
        velocitiesz[i] += accelerationsz[i] * 2.0f;
        particles.x[i] += velocitiesx[i] * 0.1f;
        particles.y[i] += velocitiesy[i] * 0.1f;
        particles.z[i] += velocitiesz[i] * 0.1f;
    }
#endif
}