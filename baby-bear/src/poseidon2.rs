//! Implementation of Poseidon2, see: https://eprint.iacr.org/2023/323
//!
//! For the diffusion matrix, 1 + Diag(V), we perform a search to find an optimized
//! vector V composed of elements with efficient multiplication algorithms in AVX2/AVX512/NEON.
//!
//! This leads to using small values (e.g. 1, 2, 3, 4) where multiplication is implemented using addition
//! and inverse powers of 2 where it is possible to avoid monty reductions.
//! Additionally, for technical reasons, having the first entry be -2 is useful.
//!
//! Optimized Diagonal for BabyBear16:
//! [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/2^27, -1/2^8, -1/16, -1/2^27].
//! Optimized Diagonal for BabyBear24:
//! [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/16, 1/2^7, 1/2^9, 1/2^27, -1/2^8, -1/4, -1/8, -1/16, -1/32, -1/64, -1/2^7, -1/2^27]
//! See poseidon2\src\diffusion.rs for information on how to double check these matrices in Sage.

use core::ops::Mul;

use p3_field::{Field, FieldAlgebra, PrimeField32};
use p3_monty_31::{
    GenericPoseidon2LinearLayersMonty31, InternalLayerBaseParameters, InternalLayerParameters,
    MontyField31, Poseidon2ExternalLayerMonty31, Poseidon2InternalLayerMonty31,
};
use p3_poseidon2::Poseidon2;
use serde::{Deserialize, Serialize};

use crate::{BabyBear, BabyBearParameters};

pub type Poseidon2InternalLayerBabyBear<const WIDTH: usize> =
    Poseidon2InternalLayerMonty31<BabyBearParameters, WIDTH, BabyBearInternalLayerParameters>;

pub type Poseidon2ExternalLayerBabyBear<const WIDTH: usize> =
    Poseidon2ExternalLayerMonty31<BabyBearParameters, WIDTH>;

/// Degree of the chosen permutation polynomial for BabyBear, used as the Poseidon2 S-Box.
///
/// As p - 1 = 15 * 2^{27} the neither 3 nor 5 satisfy gcd(p - 1, D) = 1.
/// Instead we use the next smallest available value, namely 7.
const BABYBEAR_S_BOX_DEGREE: u64 = 7;

/// An implementation of the Poseidon2 hash function specialised to run on the current architecture.
///
/// It acts on arrays of the form either `[BabyBear::Packing; WIDTH]` or `[BabyBear; WIDTH]`. For speed purposes,
/// wherever possible, input arrays should of the form `[BabyBear::Packing; WIDTH]`.
pub type Poseidon2BabyBear<const WIDTH: usize> = Poseidon2<
    <BabyBear as Field>::Packing,
    Poseidon2ExternalLayerBabyBear<WIDTH>,
    Poseidon2InternalLayerBabyBear<WIDTH>,
    WIDTH,
    BABYBEAR_S_BOX_DEGREE,
>;

/// An implementation of the the matrix multiplications in the internal and external layers of Poseidon2.
///
/// This can act on [FA; WIDTH] for any AbstractField which implements multiplication by BabyBear field elements.
/// If you have either `[BabyBear::Packing; WIDTH]` or `[BabyBear; WIDTH]` it will be much faster
/// to use `Poseidon2BabyBear<WIDTH>` instead of building a Poseidon2 permutation using this.
pub type GenericPoseidon2LinearLayersBabyBear =
    GenericPoseidon2LinearLayersMonty31<BabyBearParameters, BabyBearInternalLayerParameters>;

// In order to use BabyBear::new_array we need to convert our vector to a vector of u32's.
// To do this we make use of the fact that BabyBear::ORDER_U32 - 1 = 15 * 2^27 so for 0 <= n <= 27:
// -1/2^n = (BabyBear::ORDER_U32 - 1) >> n
// 1/2^n = -(-1/2^n) = BabyBear::ORDER_U32 - ((BabyBear::ORDER_U32 - 1) >> n)

/// The vector `[-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/2^27, -1/2^8, -1/16, -1/2^27]`
/// saved as an array of BabyBear elements.
pub const INTERNAL_DIAG_MONTY_16: [BabyBear; 16] = BabyBear::new_array([
    BabyBear::ORDER_U32 - 2,
    1,
    2,
    (BabyBear::ORDER_U32 + 1) >> 1,
    3,
    4,
    (BabyBear::ORDER_U32 - 1) >> 1,
    BabyBear::ORDER_U32 - 3,
    BabyBear::ORDER_U32 - 4,
    BabyBear::ORDER_U32 - ((BabyBear::ORDER_U32 - 1) >> 8),
    BabyBear::ORDER_U32 - ((BabyBear::ORDER_U32 - 1) >> 2),
    BabyBear::ORDER_U32 - ((BabyBear::ORDER_U32 - 1) >> 3),
    BabyBear::ORDER_U32 - 15,
    (BabyBear::ORDER_U32 - 1) >> 8,
    (BabyBear::ORDER_U32 - 1) >> 4,
    15,
]);

pub const MONTY_INVERSE: BabyBear = BabyBear::new(1);

/// The vector [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/16, 1/2^7, 1/2^9, 1/2^27, -1/2^8, -1/4, -1/8, -1/16, -1/32, -1/64, -1/2^7, -1/2^27]
/// saved as an array of BabyBear elements.
pub const INTERNAL_DIAG_MONTY_24: [BabyBear; 24] = BabyBear::new_array([
    BabyBear::ORDER_U32 - 2,
    1,
    2,
    (BabyBear::ORDER_U32 + 1) >> 1,
    3,
    4,
    (BabyBear::ORDER_U32 - 1) >> 1,
    BabyBear::ORDER_U32 - 3,
    BabyBear::ORDER_U32 - 4,
    BabyBear::ORDER_U32 - ((BabyBear::ORDER_U32 - 1) >> 8),
    BabyBear::ORDER_U32 - ((BabyBear::ORDER_U32 - 1) >> 2),
    BabyBear::ORDER_U32 - ((BabyBear::ORDER_U32 - 1) >> 3),
    BabyBear::ORDER_U32 - ((BabyBear::ORDER_U32 - 1) >> 4),
    BabyBear::ORDER_U32 - ((BabyBear::ORDER_U32 - 1) >> 7),
    BabyBear::ORDER_U32 - ((BabyBear::ORDER_U32 - 1) >> 9),
    BabyBear::ORDER_U32 - 15,
    (BabyBear::ORDER_U32 - 1) >> 8,
    (BabyBear::ORDER_U32 - 1) >> 2,
    (BabyBear::ORDER_U32 - 1) >> 3,
    (BabyBear::ORDER_U32 - 1) >> 4,
    (BabyBear::ORDER_U32 - 1) >> 5,
    (BabyBear::ORDER_U32 - 1) >> 6,
    (BabyBear::ORDER_U32 - 1) >> 7,
    15,
]);

/// Contains data needed to define the internal layers of the Poseidon2 permutation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BabyBearInternalLayerParameters;

impl InternalLayerBaseParameters<BabyBearParameters, 16> for BabyBearInternalLayerParameters {
    type ArrayLike = [MontyField31<BabyBearParameters>; 15];

    const INTERNAL_DIAG_MONTY: [BabyBear; 16] = INTERNAL_DIAG_MONTY_16;

    /// Perform the internal matrix multiplication: s -> (1 + Diag(V))s.
    /// We ignore `state[0]` as it is handled separately.
    fn internal_layer_mat_mul(
        state: &mut [MontyField31<BabyBearParameters>; 16],
        sum: MontyField31<BabyBearParameters>,
    ) {
        // The diagonal matrix is defined by the vector:
        // V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/2^27, -1/2^8, -1/16, -1/2^27]
        state[1] += sum;
        state[2] = state[2].double() + sum;
        state[3] = state[3].halve() + sum;
        state[4] = sum + state[4].double() + state[4];
        state[5] = sum + state[5].double().double();
        state[6] = sum - state[6].halve();
        state[7] = sum - (state[7].double() + state[7]);
        state[8] = sum - state[8].double().double();
        state[9] = state[9].mul_2exp_neg_n(8);
        state[9] += sum;
        state[10] = state[10].mul_2exp_neg_n(2);
        state[10] += sum;
        state[11] = state[11].mul_2exp_neg_n(3);
        state[11] += sum;
        state[12] = state[12].mul_2exp_neg_n(27);
        state[12] += sum;
        state[13] = state[13].mul_2exp_neg_n(8);
        state[13] = sum - state[13];
        state[14] = state[14].mul_2exp_neg_n(4);
        state[14] = sum - state[14];
        state[15] = state[15].mul_2exp_neg_n(27);
        state[15] = sum - state[15];
    }

    fn generic_internal_linear_layer<FA>(state: &mut [FA; 16])
    where
        FA: FieldAlgebra + Mul<BabyBear, Output = FA>,
    {
        let part_sum: FA = state[1..].iter().cloned().sum();
        let full_sum = part_sum.clone() + state[0].clone();

        // The first three diagonal elements are -2, 1, 2 so we do something custom.
        state[0] = part_sum - state[0].clone();
        state[1] = full_sum.clone() + state[1].clone();
        state[2] = full_sum.clone() + state[2].double();

        // For the remaining elements we use multiplication.
        // This could probably be improved slightly by making use of the
        // mul_2exp_u64 and div_2exp_u64 but this would involve porting div_2exp_u64 to AbstractField.
        state
            .iter_mut()
            .zip(INTERNAL_DIAG_MONTY_16)
            .skip(3)
            .for_each(|(val, diag_elem)| {
                *val = full_sum.clone() + val.clone() * diag_elem;
            });
    }
}

impl InternalLayerBaseParameters<BabyBearParameters, 24> for BabyBearInternalLayerParameters {
    type ArrayLike = [MontyField31<BabyBearParameters>; 23];

    const INTERNAL_DIAG_MONTY: [BabyBear; 24] = INTERNAL_DIAG_MONTY_24;

    /// Perform the internal matrix multiplication: s -> (1 + Diag(V))s.
    /// We ignore `state[0]` as it is handled separately.
    fn internal_layer_mat_mul(
        state: &mut [MontyField31<BabyBearParameters>; 24],
        sum: MontyField31<BabyBearParameters>,
    ) {
        // The diagonal matrix is defined by the vector:
        // V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/16, 1/2^7, 1/2^9, 1/2^27, -1/2^8, -1/4, -1/8, -1/16, -1/32, -1/64, -1/2^7, -1/2^27]
        state[1] += sum;
        state[2] = state[2].double() + sum;
        state[3] = state[3].halve() + sum;
        state[4] = sum + state[4].double() + state[4];
        state[5] = sum + state[5].double().double();
        state[6] = sum - state[6].halve();
        state[7] = sum - (state[7].double() + state[7]);
        state[8] = sum - state[8].double().double();
        state[9] = state[9].mul_2exp_neg_n(8);
        state[9] += sum;
        state[10] = state[10].mul_2exp_neg_n(2);
        state[10] += sum;
        state[11] = state[11].mul_2exp_neg_n(3);
        state[11] += sum;
        state[12] = state[12].mul_2exp_neg_n(4);
        state[12] += sum;
        state[13] = state[13].mul_2exp_neg_n(7);
        state[13] += sum;
        state[14] = state[14].mul_2exp_neg_n(9);
        state[14] += sum;
        state[15] = state[15].mul_2exp_neg_n(27);
        state[15] += sum;
        state[16] = state[16].mul_2exp_neg_n(8);
        state[16] = sum - state[16];
        state[17] = state[17].mul_2exp_neg_n(2);
        state[17] = sum - state[17];
        state[18] = state[18].mul_2exp_neg_n(3);
        state[18] = sum - state[18];
        state[19] = state[19].mul_2exp_neg_n(4);
        state[19] = sum - state[19];
        state[20] = state[20].mul_2exp_neg_n(5);
        state[20] = sum - state[20];
        state[21] = state[21].mul_2exp_neg_n(6);
        state[21] = sum - state[21];
        state[22] = state[22].mul_2exp_neg_n(7);
        state[22] = sum - state[22];
        state[23] = state[23].mul_2exp_neg_n(27);
        state[23] = sum - state[23];
    }

    fn generic_internal_linear_layer<FA>(state: &mut [FA; 24])
    where
        FA: FieldAlgebra + Mul<BabyBear, Output = FA>,
    {
        let part_sum: FA = state[1..].iter().cloned().sum();
        let full_sum = part_sum.clone() + state[0].clone();

        // The first three diagonal elements are -2, 1, 2 so we do something custom.
        state[0] = part_sum - state[0].clone();
        state[1] = full_sum.clone() + state[1].clone();
        state[2] = full_sum.clone() + state[2].double();

        // For the remaining elements we use multiplication.
        // This could probably be improved slightly by making use of the
        // mul_2exp_u64 and div_2exp_u64 but this would involve porting div_2exp_u64 to AbstractField.
        state
            .iter_mut()
            .zip(INTERNAL_DIAG_MONTY_24)
            .skip(3)
            .for_each(|(val, diag_elem)| {
                *val = full_sum.clone() + val.clone() * diag_elem;
            });
    }
}

impl InternalLayerParameters<BabyBearParameters, 16> for BabyBearInternalLayerParameters {}
impl InternalLayerParameters<BabyBearParameters, 24> for BabyBearInternalLayerParameters {}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use p3_field::FieldAlgebra;
    use p3_poseidon2::ExternalLayerConstants;
    use p3_symmetric::Permutation;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoroshiro128Plus;

    use super::*;

    type F = BabyBear;

    // We need to make some round constants. We use Xoroshiro128Plus for this as we can easily match this PRNG in sage.
    // See: https://github.com/0xPolygonZero/hash-constants for the sage code used to create all these tests.

    /// Test on a roughly random input.
    /// This random input is generated by the following sage code:
    /// set_random_seed(16)
    /// vector([BB.random_element() for t in range(16)]).
    #[test]
    fn test_poseidon2_width_16_random() {
        let mut input: [F; 16] = [
            894848333, 1437655012, 1200606629, 1690012884, 71131202, 1749206695, 1717947831,
            120589055, 19776022, 42382981, 1831865506, 724844064, 171220207, 1299207443, 227047920,
            1783754913,
        ]
        .map(F::from_canonical_u32);

        let expected: [F; 16] = [
            1255099308, 941729227, 93609187, 112406640, 492658670, 1824768948, 812517469,
            1055381989, 670973674, 1407235524, 891397172, 1003245378, 1381303998, 1564172645,
            1399931635, 1005462965,
        ]
        .map(F::from_canonical_u32);

        let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        let perm = Poseidon2BabyBear::new_from_rng_128(&mut rng);

        perm.permute_mut(&mut input);
        assert_eq!(input, expected);
    }

    #[test]
    fn test_poseidon2_width16_0() {
        let rc: [[BabyBear; 16]; 30] = [
            [
                BabyBear::from_wrapped_u32(2110014213),
                BabyBear::from_wrapped_u32(3964964605),
                BabyBear::from_wrapped_u32(2190662774),
                BabyBear::from_wrapped_u32(2732996483),
                BabyBear::from_wrapped_u32(640767983),
                BabyBear::from_wrapped_u32(3403899136),
                BabyBear::from_wrapped_u32(1716033721),
                BabyBear::from_wrapped_u32(1606702601),
                BabyBear::from_wrapped_u32(3759873288),
                BabyBear::from_wrapped_u32(1466015491),
                BabyBear::from_wrapped_u32(1498308946),
                BabyBear::from_wrapped_u32(2844375094),
                BabyBear::from_wrapped_u32(3042463841),
                BabyBear::from_wrapped_u32(1969905919),
                BabyBear::from_wrapped_u32(4109944726),
                BabyBear::from_wrapped_u32(3925048366),
            ],
            [
                BabyBear::from_wrapped_u32(3706859504),
                BabyBear::from_wrapped_u32(759122502),
                BabyBear::from_wrapped_u32(3167665446),
                BabyBear::from_wrapped_u32(1131812921),
                BabyBear::from_wrapped_u32(1080754908),
                BabyBear::from_wrapped_u32(4080114493),
                BabyBear::from_wrapped_u32(893583089),
                BabyBear::from_wrapped_u32(2019677373),
                BabyBear::from_wrapped_u32(3128604556),
                BabyBear::from_wrapped_u32(580640471),
                BabyBear::from_wrapped_u32(3277620260),
                BabyBear::from_wrapped_u32(842931656),
                BabyBear::from_wrapped_u32(548879852),
                BabyBear::from_wrapped_u32(3608554714),
                BabyBear::from_wrapped_u32(3575647916),
                BabyBear::from_wrapped_u32(81826002),
            ],
            [
                BabyBear::from_wrapped_u32(4289086263),
                BabyBear::from_wrapped_u32(1563933798),
                BabyBear::from_wrapped_u32(1440025885),
                BabyBear::from_wrapped_u32(184445025),
                BabyBear::from_wrapped_u32(2598651360),
                BabyBear::from_wrapped_u32(1396647410),
                BabyBear::from_wrapped_u32(1575877922),
                BabyBear::from_wrapped_u32(3303853401),
                BabyBear::from_wrapped_u32(137125468),
                BabyBear::from_wrapped_u32(765010148),
                BabyBear::from_wrapped_u32(633675867),
                BabyBear::from_wrapped_u32(2037803363),
                BabyBear::from_wrapped_u32(2573389828),
                BabyBear::from_wrapped_u32(1895729703),
                BabyBear::from_wrapped_u32(541515871),
                BabyBear::from_wrapped_u32(1783382863),
            ],
            [
                BabyBear::from_wrapped_u32(2641856484),
                BabyBear::from_wrapped_u32(3035743342),
                BabyBear::from_wrapped_u32(3672796326),
                BabyBear::from_wrapped_u32(245668751),
                BabyBear::from_wrapped_u32(2025460432),
                BabyBear::from_wrapped_u32(201609705),
                BabyBear::from_wrapped_u32(286217151),
                BabyBear::from_wrapped_u32(4093475563),
                BabyBear::from_wrapped_u32(2519572182),
                BabyBear::from_wrapped_u32(3080699870),
                BabyBear::from_wrapped_u32(2762001832),
                BabyBear::from_wrapped_u32(1244250808),
                BabyBear::from_wrapped_u32(606038199),
                BabyBear::from_wrapped_u32(3182740831),
                BabyBear::from_wrapped_u32(73007766),
                BabyBear::from_wrapped_u32(2572204153),
            ],
            [
                BabyBear::from_wrapped_u32(1196780786),
                BabyBear::from_wrapped_u32(3447394443),
                BabyBear::from_wrapped_u32(747167305),
                BabyBear::from_wrapped_u32(2968073607),
                BabyBear::from_wrapped_u32(1053214930),
                BabyBear::from_wrapped_u32(1074411832),
                BabyBear::from_wrapped_u32(4016794508),
                BabyBear::from_wrapped_u32(1570312929),
                BabyBear::from_wrapped_u32(113576933),
                BabyBear::from_wrapped_u32(4042581186),
                BabyBear::from_wrapped_u32(3634515733),
                BabyBear::from_wrapped_u32(1032701597),
                BabyBear::from_wrapped_u32(2364839308),
                BabyBear::from_wrapped_u32(3840286918),
                BabyBear::from_wrapped_u32(888378655),
                BabyBear::from_wrapped_u32(2520191583),
            ],
            [
                BabyBear::from_wrapped_u32(36046858),
                BabyBear::from_wrapped_u32(2927525953),
                BabyBear::from_wrapped_u32(3912129105),
                BabyBear::from_wrapped_u32(4004832531),
                BabyBear::from_wrapped_u32(193772436),
                BabyBear::from_wrapped_u32(1590247392),
                BabyBear::from_wrapped_u32(4125818172),
                BabyBear::from_wrapped_u32(2516251696),
                BabyBear::from_wrapped_u32(4050945750),
                BabyBear::from_wrapped_u32(269498914),
                BabyBear::from_wrapped_u32(1973292656),
                BabyBear::from_wrapped_u32(891403491),
                BabyBear::from_wrapped_u32(1845429189),
                BabyBear::from_wrapped_u32(2611996363),
                BabyBear::from_wrapped_u32(2310542653),
                BabyBear::from_wrapped_u32(4071195740),
            ],
            [
                BabyBear::from_wrapped_u32(3505307391),
                BabyBear::from_wrapped_u32(786445290),
                BabyBear::from_wrapped_u32(3815313971),
                BabyBear::from_wrapped_u32(1111591756),
                BabyBear::from_wrapped_u32(4233279834),
                BabyBear::from_wrapped_u32(2775453034),
                BabyBear::from_wrapped_u32(1991257625),
                BabyBear::from_wrapped_u32(2940505809),
                BabyBear::from_wrapped_u32(2751316206),
                BabyBear::from_wrapped_u32(1028870679),
                BabyBear::from_wrapped_u32(1282466273),
                BabyBear::from_wrapped_u32(1059053371),
                BabyBear::from_wrapped_u32(834521354),
                BabyBear::from_wrapped_u32(138721483),
                BabyBear::from_wrapped_u32(3100410803),
                BabyBear::from_wrapped_u32(3843128331),
            ],
            [
                BabyBear::from_wrapped_u32(3878220780),
                BabyBear::from_wrapped_u32(4058162439),
                BabyBear::from_wrapped_u32(1478942487),
                BabyBear::from_wrapped_u32(799012923),
                BabyBear::from_wrapped_u32(496734827),
                BabyBear::from_wrapped_u32(3521261236),
                BabyBear::from_wrapped_u32(755421082),
                BabyBear::from_wrapped_u32(1361409515),
                BabyBear::from_wrapped_u32(392099473),
                BabyBear::from_wrapped_u32(3178453393),
                BabyBear::from_wrapped_u32(4068463721),
                BabyBear::from_wrapped_u32(7935614),
                BabyBear::from_wrapped_u32(4140885645),
                BabyBear::from_wrapped_u32(2150748066),
                BabyBear::from_wrapped_u32(1685210312),
                BabyBear::from_wrapped_u32(3852983224),
            ],
            [
                BabyBear::from_wrapped_u32(2896943075),
                BabyBear::from_wrapped_u32(3087590927),
                BabyBear::from_wrapped_u32(992175959),
                BabyBear::from_wrapped_u32(970216228),
                BabyBear::from_wrapped_u32(3473630090),
                BabyBear::from_wrapped_u32(3899670400),
                BabyBear::from_wrapped_u32(3603388822),
                BabyBear::from_wrapped_u32(2633488197),
                BabyBear::from_wrapped_u32(2479406964),
                BabyBear::from_wrapped_u32(2420952999),
                BabyBear::from_wrapped_u32(1852516800),
                BabyBear::from_wrapped_u32(4253075697),
                BabyBear::from_wrapped_u32(979699862),
                BabyBear::from_wrapped_u32(1163403191),
                BabyBear::from_wrapped_u32(1608599874),
                BabyBear::from_wrapped_u32(3056104448),
            ],
            [
                BabyBear::from_wrapped_u32(3779109343),
                BabyBear::from_wrapped_u32(536205958),
                BabyBear::from_wrapped_u32(4183458361),
                BabyBear::from_wrapped_u32(1649720295),
                BabyBear::from_wrapped_u32(1444912244),
                BabyBear::from_wrapped_u32(3122230878),
                BabyBear::from_wrapped_u32(384301396),
                BabyBear::from_wrapped_u32(4228198516),
                BabyBear::from_wrapped_u32(1662916865),
                BabyBear::from_wrapped_u32(4082161114),
                BabyBear::from_wrapped_u32(2121897314),
                BabyBear::from_wrapped_u32(1706239958),
                BabyBear::from_wrapped_u32(4166959388),
                BabyBear::from_wrapped_u32(1626054781),
                BabyBear::from_wrapped_u32(3005858978),
                BabyBear::from_wrapped_u32(1431907253),
            ],
            [
                BabyBear::from_wrapped_u32(1418914503),
                BabyBear::from_wrapped_u32(1365856753),
                BabyBear::from_wrapped_u32(3942715745),
                BabyBear::from_wrapped_u32(1429155552),
                BabyBear::from_wrapped_u32(3545642795),
                BabyBear::from_wrapped_u32(3772474257),
                BabyBear::from_wrapped_u32(1621094396),
                BabyBear::from_wrapped_u32(2154399145),
                BabyBear::from_wrapped_u32(826697382),
                BabyBear::from_wrapped_u32(1700781391),
                BabyBear::from_wrapped_u32(3539164324),
                BabyBear::from_wrapped_u32(652815039),
                BabyBear::from_wrapped_u32(442484755),
                BabyBear::from_wrapped_u32(2055299391),
                BabyBear::from_wrapped_u32(1064289978),
                BabyBear::from_wrapped_u32(1152335780),
            ],
            [
                BabyBear::from_wrapped_u32(3417648695),
                BabyBear::from_wrapped_u32(186040114),
                BabyBear::from_wrapped_u32(3475580573),
                BabyBear::from_wrapped_u32(2113941250),
                BabyBear::from_wrapped_u32(1779573826),
                BabyBear::from_wrapped_u32(1573808590),
                BabyBear::from_wrapped_u32(3235694804),
                BabyBear::from_wrapped_u32(2922195281),
                BabyBear::from_wrapped_u32(1119462702),
                BabyBear::from_wrapped_u32(3688305521),
                BabyBear::from_wrapped_u32(1849567013),
                BabyBear::from_wrapped_u32(667446787),
                BabyBear::from_wrapped_u32(753897224),
                BabyBear::from_wrapped_u32(1896396780),
                BabyBear::from_wrapped_u32(3143026334),
                BabyBear::from_wrapped_u32(3829603876),
            ],
            [
                BabyBear::from_wrapped_u32(859661334),
                BabyBear::from_wrapped_u32(3898844357),
                BabyBear::from_wrapped_u32(180258337),
                BabyBear::from_wrapped_u32(2321867017),
                BabyBear::from_wrapped_u32(3599002504),
                BabyBear::from_wrapped_u32(2886782421),
                BabyBear::from_wrapped_u32(3038299378),
                BabyBear::from_wrapped_u32(1035366250),
                BabyBear::from_wrapped_u32(2038912197),
                BabyBear::from_wrapped_u32(2920174523),
                BabyBear::from_wrapped_u32(1277696101),
                BabyBear::from_wrapped_u32(2785700290),
                BabyBear::from_wrapped_u32(3806504335),
                BabyBear::from_wrapped_u32(3518858933),
                BabyBear::from_wrapped_u32(654843672),
                BabyBear::from_wrapped_u32(2127120275),
            ],
            [
                BabyBear::from_wrapped_u32(1548195514),
                BabyBear::from_wrapped_u32(2378056027),
                BabyBear::from_wrapped_u32(390914568),
                BabyBear::from_wrapped_u32(1472049779),
                BabyBear::from_wrapped_u32(1552596765),
                BabyBear::from_wrapped_u32(1905886441),
                BabyBear::from_wrapped_u32(1611959354),
                BabyBear::from_wrapped_u32(3653263304),
                BabyBear::from_wrapped_u32(3423946386),
                BabyBear::from_wrapped_u32(340857935),
                BabyBear::from_wrapped_u32(2208879480),
                BabyBear::from_wrapped_u32(139364268),
                BabyBear::from_wrapped_u32(3447281773),
                BabyBear::from_wrapped_u32(3777813707),
                BabyBear::from_wrapped_u32(55640413),
                BabyBear::from_wrapped_u32(4101901741),
            ],
            [
                BabyBear::from_wrapped_u32(104929687),
                BabyBear::from_wrapped_u32(1459980974),
                BabyBear::from_wrapped_u32(1831234737),
                BabyBear::from_wrapped_u32(457139004),
                BabyBear::from_wrapped_u32(2581487628),
                BabyBear::from_wrapped_u32(2112044563),
                BabyBear::from_wrapped_u32(3567013861),
                BabyBear::from_wrapped_u32(2792004347),
                BabyBear::from_wrapped_u32(576325418),
                BabyBear::from_wrapped_u32(41126132),
                BabyBear::from_wrapped_u32(2713562324),
                BabyBear::from_wrapped_u32(151213722),
                BabyBear::from_wrapped_u32(2891185935),
                BabyBear::from_wrapped_u32(546846420),
                BabyBear::from_wrapped_u32(2939794919),
                BabyBear::from_wrapped_u32(2543469905)
            ],
            [
                BabyBear::from_wrapped_u32(2191909784),
                BabyBear::from_wrapped_u32(3315138460),
                BabyBear::from_wrapped_u32(530414574),
                BabyBear::from_wrapped_u32(1242280418),
                BabyBear::from_wrapped_u32(1211740715),
                BabyBear::from_wrapped_u32(3993672165),
                BabyBear::from_wrapped_u32(2505083323),
                BabyBear::from_wrapped_u32(3845798801),
                BabyBear::from_wrapped_u32(538768466),
                BabyBear::from_wrapped_u32(2063567560),
                BabyBear::from_wrapped_u32(3366148274),
                BabyBear::from_wrapped_u32(1449831887),
                BabyBear::from_wrapped_u32(2408012466),
                BabyBear::from_wrapped_u32(294726285),
                BabyBear::from_wrapped_u32(3943435493),
                BabyBear::from_wrapped_u32(924016661),
            ],
            [
                BabyBear::from_wrapped_u32(3633138367),
                BabyBear::from_wrapped_u32(3222789372),
                BabyBear::from_wrapped_u32(809116305),
                BabyBear::from_wrapped_u32(30100013),
                BabyBear::from_wrapped_u32(2655172876),
                BabyBear::from_wrapped_u32(2564247117),
                BabyBear::from_wrapped_u32(2478649732),
                BabyBear::from_wrapped_u32(4113689151),
                BabyBear::from_wrapped_u32(4120146082),
                BabyBear::from_wrapped_u32(2512308515),
                BabyBear::from_wrapped_u32(650406041),
                BabyBear::from_wrapped_u32(4240012393),
                BabyBear::from_wrapped_u32(2683508708),
                BabyBear::from_wrapped_u32(951073977),
                BabyBear::from_wrapped_u32(3460081988),
                BabyBear::from_wrapped_u32(339124269),
            ],
            [
                BabyBear::from_wrapped_u32(130182653),
                BabyBear::from_wrapped_u32(2755946749),
                BabyBear::from_wrapped_u32(542600513),
                BabyBear::from_wrapped_u32(2816103022),
                BabyBear::from_wrapped_u32(1931786340),
                BabyBear::from_wrapped_u32(2044470840),
                BabyBear::from_wrapped_u32(1709908013),
                BabyBear::from_wrapped_u32(2938369043),
                BabyBear::from_wrapped_u32(3640399693),
                BabyBear::from_wrapped_u32(1374470239),
                BabyBear::from_wrapped_u32(2191149676),
                BabyBear::from_wrapped_u32(2637495682),
                BabyBear::from_wrapped_u32(4236394040),
                BabyBear::from_wrapped_u32(2289358846),
                BabyBear::from_wrapped_u32(3833368530),
                BabyBear::from_wrapped_u32(974546524),
            ],
            [
                BabyBear::from_wrapped_u32(3306659113),
                BabyBear::from_wrapped_u32(2234814261),
                BabyBear::from_wrapped_u32(1188782305),
                BabyBear::from_wrapped_u32(223782844),
                BabyBear::from_wrapped_u32(2248980567),
                BabyBear::from_wrapped_u32(2309786141),
                BabyBear::from_wrapped_u32(2023401627),
                BabyBear::from_wrapped_u32(3278877413),
                BabyBear::from_wrapped_u32(2022138149),
                BabyBear::from_wrapped_u32(575851471),
                BabyBear::from_wrapped_u32(1612560780),
                BabyBear::from_wrapped_u32(3926656936),
                BabyBear::from_wrapped_u32(3318548977),
                BabyBear::from_wrapped_u32(2591863678),
                BabyBear::from_wrapped_u32(188109355),
                BabyBear::from_wrapped_u32(4217723909),
            ],
            [
                BabyBear::from_wrapped_u32(1564209905),
                BabyBear::from_wrapped_u32(2154197895),
                BabyBear::from_wrapped_u32(2459687029),
                BabyBear::from_wrapped_u32(2870634489),
                BabyBear::from_wrapped_u32(1375012945),
                BabyBear::from_wrapped_u32(1529454825),
                BabyBear::from_wrapped_u32(306140690),
                BabyBear::from_wrapped_u32(2855578299),
                BabyBear::from_wrapped_u32(1246997295),
                BabyBear::from_wrapped_u32(3024298763),
                BabyBear::from_wrapped_u32(1915270363),
                BabyBear::from_wrapped_u32(1218245412),
                BabyBear::from_wrapped_u32(2479314020),
                BabyBear::from_wrapped_u32(2989827755),
                BabyBear::from_wrapped_u32(814378556),
                BabyBear::from_wrapped_u32(4039775921),
            ],
            [
                BabyBear::from_wrapped_u32(1165280628),
                BabyBear::from_wrapped_u32(1203983801),
                BabyBear::from_wrapped_u32(3814740033),
                BabyBear::from_wrapped_u32(1919627044),
                BabyBear::from_wrapped_u32(600240215),
                BabyBear::from_wrapped_u32(773269071),
                BabyBear::from_wrapped_u32(486685186),
                BabyBear::from_wrapped_u32(4254048810),
                BabyBear::from_wrapped_u32(1415023565),
                BabyBear::from_wrapped_u32(502840102),
                BabyBear::from_wrapped_u32(4225648358),
                BabyBear::from_wrapped_u32(510217063),
                BabyBear::from_wrapped_u32(166444818),
                BabyBear::from_wrapped_u32(1430745893),
                BabyBear::from_wrapped_u32(1376516190),
                BabyBear::from_wrapped_u32(1775891321),
            ],
            [
                BabyBear::from_wrapped_u32(1170945922),
                BabyBear::from_wrapped_u32(1105391877),
                BabyBear::from_wrapped_u32(261536467),
                BabyBear::from_wrapped_u32(1401687994),
                BabyBear::from_wrapped_u32(1022529847),
                BabyBear::from_wrapped_u32(2476446456),
                BabyBear::from_wrapped_u32(2603844878),
                BabyBear::from_wrapped_u32(3706336043),
                BabyBear::from_wrapped_u32(3463053714),
                BabyBear::from_wrapped_u32(1509644517),
                BabyBear::from_wrapped_u32(588552318),
                BabyBear::from_wrapped_u32(65252581),
                BabyBear::from_wrapped_u32(3696502656),
                BabyBear::from_wrapped_u32(2183330763),
                BabyBear::from_wrapped_u32(3664021233),
                BabyBear::from_wrapped_u32(1643809916),
            ],
            [
                BabyBear::from_wrapped_u32(2922875898),
                BabyBear::from_wrapped_u32(3740690643),
                BabyBear::from_wrapped_u32(3932461140),
                BabyBear::from_wrapped_u32(161156271),
                BabyBear::from_wrapped_u32(2619943483),
                BabyBear::from_wrapped_u32(4077039509),
                BabyBear::from_wrapped_u32(2921201703),
                BabyBear::from_wrapped_u32(2085619718),
                BabyBear::from_wrapped_u32(2065264646),
                BabyBear::from_wrapped_u32(2615693812),
                BabyBear::from_wrapped_u32(3116555433),
                BabyBear::from_wrapped_u32(246100007),
                BabyBear::from_wrapped_u32(4281387154),
                BabyBear::from_wrapped_u32(4046141001),
                BabyBear::from_wrapped_u32(4027749321),
                BabyBear::from_wrapped_u32(111611860),
            ],
            [
                BabyBear::from_wrapped_u32(2066954820),
                BabyBear::from_wrapped_u32(2502099969),
                BabyBear::from_wrapped_u32(2915053115),
                BabyBear::from_wrapped_u32(2362518586),
                BabyBear::from_wrapped_u32(366091708),
                BabyBear::from_wrapped_u32(2083204932),
                BabyBear::from_wrapped_u32(4138385632),
                BabyBear::from_wrapped_u32(3195157567),
                BabyBear::from_wrapped_u32(1318086382),
                BabyBear::from_wrapped_u32(521723799),
                BabyBear::from_wrapped_u32(702443405),
                BabyBear::from_wrapped_u32(2507670985),
                BabyBear::from_wrapped_u32(1760347557),
                BabyBear::from_wrapped_u32(2631999893),
                BabyBear::from_wrapped_u32(1672737554),
                BabyBear::from_wrapped_u32(1060867760),
            ],
            [
                BabyBear::from_wrapped_u32(2359801781),
                BabyBear::from_wrapped_u32(2800231467),
                BabyBear::from_wrapped_u32(3010357035),
                BabyBear::from_wrapped_u32(1035997899),
                BabyBear::from_wrapped_u32(1210110952),
                BabyBear::from_wrapped_u32(1018506770),
                BabyBear::from_wrapped_u32(2799468177),
                BabyBear::from_wrapped_u32(1479380761),
                BabyBear::from_wrapped_u32(1536021911),
                BabyBear::from_wrapped_u32(358993854),
                BabyBear::from_wrapped_u32(579904113),
                BabyBear::from_wrapped_u32(3432144800),
                BabyBear::from_wrapped_u32(3625515809),
                BabyBear::from_wrapped_u32(199241497),
                BabyBear::from_wrapped_u32(4058304109),
                BabyBear::from_wrapped_u32(2590164234),
            ],
            [
                BabyBear::from_wrapped_u32(1688530738),
                BabyBear::from_wrapped_u32(1580733335),
                BabyBear::from_wrapped_u32(2443981517),
                BabyBear::from_wrapped_u32(2206270565),
                BabyBear::from_wrapped_u32(2780074229),
                BabyBear::from_wrapped_u32(2628739677),
                BabyBear::from_wrapped_u32(2940123659),
                BabyBear::from_wrapped_u32(4145206827),
                BabyBear::from_wrapped_u32(3572278009),
                BabyBear::from_wrapped_u32(2779607509),
                BabyBear::from_wrapped_u32(1098718697),
                BabyBear::from_wrapped_u32(1424913749),
                BabyBear::from_wrapped_u32(2224415875),
                BabyBear::from_wrapped_u32(1108922178),
                BabyBear::from_wrapped_u32(3646272562),
                BabyBear::from_wrapped_u32(3935186184),
            ],
            [
                BabyBear::from_wrapped_u32(820046587),
                BabyBear::from_wrapped_u32(1393386250),
                BabyBear::from_wrapped_u32(2665818575),
                BabyBear::from_wrapped_u32(2231782019),
                BabyBear::from_wrapped_u32(672377010),
                BabyBear::from_wrapped_u32(1920315467),
                BabyBear::from_wrapped_u32(1913164407),
                BabyBear::from_wrapped_u32(2029526876),
                BabyBear::from_wrapped_u32(2629271820),
                BabyBear::from_wrapped_u32(384320012),
                BabyBear::from_wrapped_u32(4112320585),
                BabyBear::from_wrapped_u32(3131824773),
                BabyBear::from_wrapped_u32(2347818197),
                BabyBear::from_wrapped_u32(2220997386),
                BabyBear::from_wrapped_u32(1772368609),
                BabyBear::from_wrapped_u32(2579960095),
            ],
            [
                BabyBear::from_wrapped_u32(3544930873),
                BabyBear::from_wrapped_u32(225847443),
                BabyBear::from_wrapped_u32(3070082278),
                BabyBear::from_wrapped_u32(95643305),
                BabyBear::from_wrapped_u32(3438572042),
                BabyBear::from_wrapped_u32(3312856509),
                BabyBear::from_wrapped_u32(615850007),
                BabyBear::from_wrapped_u32(1863868773),
                BabyBear::from_wrapped_u32(803582265),
                BabyBear::from_wrapped_u32(3461976859),
                BabyBear::from_wrapped_u32(2903025799),
                BabyBear::from_wrapped_u32(1482092434),
                BabyBear::from_wrapped_u32(3902972499),
                BabyBear::from_wrapped_u32(3872341868),
                BabyBear::from_wrapped_u32(1530411808),
                BabyBear::from_wrapped_u32(2214923584),
            ],
            [
                BabyBear::from_wrapped_u32(3118792481),
                BabyBear::from_wrapped_u32(2241076515),
                BabyBear::from_wrapped_u32(3983669831),
                BabyBear::from_wrapped_u32(3180915147),
                BabyBear::from_wrapped_u32(3838626501),
                BabyBear::from_wrapped_u32(1921630011),
                BabyBear::from_wrapped_u32(3415351771),
                BabyBear::from_wrapped_u32(2249953859),
                BabyBear::from_wrapped_u32(3755081630),
                BabyBear::from_wrapped_u32(486327260),
                BabyBear::from_wrapped_u32(1227575720),
                BabyBear::from_wrapped_u32(3643869379),
                BabyBear::from_wrapped_u32(2982026073),
                BabyBear::from_wrapped_u32(2466043731),
                BabyBear::from_wrapped_u32(1982634375),
                BabyBear::from_wrapped_u32(3769609014),
            ],
            [
                BabyBear::from_wrapped_u32(2195455495),
                BabyBear::from_wrapped_u32(2596863283),
                BabyBear::from_wrapped_u32(4244994973),
                BabyBear::from_wrapped_u32(1983609348),
                BabyBear::from_wrapped_u32(4019674395),
                BabyBear::from_wrapped_u32(3469982031),
                BabyBear::from_wrapped_u32(1458697570),
                BabyBear::from_wrapped_u32(1593516217),
                BabyBear::from_wrapped_u32(1963896497),
                BabyBear::from_wrapped_u32(3115309118),
                BabyBear::from_wrapped_u32(1659132465),
                BabyBear::from_wrapped_u32(2536770756),
                BabyBear::from_wrapped_u32(3059294171),
                BabyBear::from_wrapped_u32(2618031334),
                BabyBear::from_wrapped_u32(2040903247),
                BabyBear::from_wrapped_u32(3799795076),
            ]
        ];


        let mut input: [F; 16] = [0; 16].map(F::from_canonical_u32);
        let expected: [F; 16] = [0; 16].map(F::from_canonical_u32);

        // let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        // let perm = Poseidon2BabyBear::new_from_rng_128(&mut rng);

        let mut round_constants = rc.to_vec();
        let internal_start = 8 / 2;
        let internal_end = (8 / 2) + 13;
        let internal_round_constants =
            round_constants.drain(internal_start..internal_end).map(|vec| vec[0]).collect::<Vec<_>>();
        let external_round_constants = round_constants;
        let initial_constants = external_round_constants[0..8 / 2].to_vec();
        let terminal_constants = external_round_constants[8 / 2..8].to_vec();
        let external_layer_constants =
            ExternalLayerConstants::<BabyBear, 16>::new(initial_constants, terminal_constants);
        let perm = Poseidon2BabyBear::new(external_layer_constants, internal_round_constants);
        perm.permute_mut(&mut input);
        // println!("{:?}", input);
        assert_eq!(input, expected);
    }



    /// Test on a roughly random input.
    /// This random input is generated by the following sage code:
    /// set_random_seed(24)
    /// vector([BB.random_element() for t in range(24)]).
    #[test]
    fn test_poseidon2_width_24_random() {
        let mut input: [F; 24] = [
            886409618, 1327899896, 1902407911, 591953491, 648428576, 1844789031, 1198336108,
            355597330, 1799586834, 59617783, 790334801, 1968791836, 559272107, 31054313,
            1042221543, 474748436, 135686258, 263665994, 1962340735, 1741539604, 449439011,
            1131357108, 50869465, 1589724894,
        ]
        .map(F::from_canonical_u32);

        let expected: [F; 24] = [
            249424342, 562262148, 757431114, 354243402, 57767055, 976981973, 1393169022,
            1774550827, 1527742125, 1019514605, 1776327602, 266236737, 1412355182, 1070239213,
            426390978, 1775539440, 1527732214, 1101406020, 1417710778, 1699632661, 413672313,
            820348291, 1067197851, 1669055675,
        ]
        .map(F::from_canonical_u32);

        let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        let perm = Poseidon2BabyBear::new_from_rng_128(&mut rng);

        perm.permute_mut(&mut input);

        assert_eq!(input, expected);
    }

    /// Test the generic internal layer against the optimized internal layer
    /// for a random input of width 16.
    #[test]
    fn test_generic_internal_linear_layer_16() {
        let mut rng = rand::thread_rng();
        let mut input1: [F; 16] = rng.gen();
        let mut input2 = input1;

        let part_sum: F = input1[1..].iter().cloned().sum();
        let full_sum = part_sum + input1[0];

        input1[0] = part_sum - input1[0];

        BabyBearInternalLayerParameters::internal_layer_mat_mul(&mut input1, full_sum);
        BabyBearInternalLayerParameters::generic_internal_linear_layer(&mut input2);

        assert_eq!(input1, input2);
    }

    /// Test the generic internal layer against the optimized internal layer
    /// for a random input of width 24.
    #[test]
    fn test_generic_internal_linear_layer_24() {
        let mut rng = rand::thread_rng();
        let mut input1: [F; 24] = rng.gen();
        let mut input2 = input1;

        let part_sum: F = input1[1..].iter().cloned().sum();
        let full_sum = part_sum + input1[0];

        input1[0] = part_sum - input1[0];

        BabyBearInternalLayerParameters::internal_layer_mat_mul(&mut input1, full_sum);
        BabyBearInternalLayerParameters::generic_internal_linear_layer(&mut input2);

        assert_eq!(input1, input2);
    }
}
