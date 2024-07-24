use dep::bignum::BigNum;
use dep::bignum::BigNumTrait;
use dep::bignum::BigNumInstanceTrait;
use crate::scalar_field::ScalarField;
use crate::CurveParamsTrait;
use crate::BigCurve;
/**
 * @brief CurveJ represents a Short Weierstrass elliptic curve using Jacobian coordinates.
 *        representation in Jacobian form is X, Y, Z
 *        representation in affine form is x, y
 *        relation between both is: x = X / Z^2  , y = Y / Z^3
 *
 * @description The purpose of this class is to speed up witness generation when optimally constraining ecc operations
 *              When constructing constraints, we want to work in AFFINE coordinates,
 *              because of the smaller number of field multiplications (3 muls vs ~14 for Jacobian)
 *              HOWEVER, when working in affine coordinates, 1 modular inverse must be computed per group operation.
 *              This is VERY TIME CONSUMING TO DO when generating witnesses, as this is a modular inverse over a non-native field.
 *              This costs approximately 2D multiplications, where D is the bit-width of the curve field.
 *
 *              To solve this problem, we perform the ENTIRE COMPUTATION in unconstrained functions using JACOBIAN coordiantes
 *              (which don't require inverses). We record each ECC op performed in a JTranscript object
 *              We then compute a Montgomery Batch Inverse to compute ALL of the inverses we require when evaluating using affine arithmetic.
 *              i.e. we compute 1 modular inverse instead of ~256 or 320 depending on the elliptic curve.
 *              Yes, this is an extremely complex solution to a simple problem. Such is life. Inverses are expensive to generate witnesses for.
 **/
struct CurveJ<BigNumInstance, BigNum, CurveParams> {
    x: BigNum,
    y: BigNum,
    z: BigNum,
    is_infinity: bool
}

/**
 * @brief A transcript of a group operation in Jacobian coordinates
 * x3, y3, z3 = the output of the group operation
 * lambda_numerator = numerator of the `lambda` term (the denominator is assumed to be z3)
 **/
struct JTranscript<BigNumInstance, BigNum> {
    lambda_numerator: BigNum,
    x3: BigNum,
    y3: BigNum,
    z3: BigNum
}

impl<BigNumInstance, BigNum> JTranscript<BigNumInstance, BigNum> where BigNum: BigNumTrait, BigNumInstance: BigNumInstanceTrait<BigNum> {
    unconstrained fn new() -> Self {
        JTranscript { lambda_numerator: BigNum::new(), x3: BigNum::new(), y3: BigNum::new(), z3: BigNum::new() }
    }
}

/**
 * @brief A transcript of a group operation in Affine coordinates
 * x3, y3 = the output of the group operation
 * lambda = gradient of the line that passes through input points of group operation
 * For addition, lambda = (y2 - y1) / (x2 - x1)
 * For doubling, lambda = (3 * x1 * x1) / (2 * y1)
 * If we have an array of JTranscript objects, we can turn them into AffineTranscript objects with only 1 modular inverse
 **/
struct AffineTranscript<BigNumInstance, BigNum> {
    lambda: BigNum,
    x3: BigNum,
    y3: BigNum
}

/**
 * @brief construct a sequence of AffineTranscript objects from a sequence of Jacobian transcript objects
 **/
impl<BigNumInstance, BigNum> AffineTranscript<BigNumInstance, BigNum> where BigNum: BigNumTrait, BigNumInstance: BigNumInstanceTrait<BigNum> {
    fn new() -> Self {
        AffineTranscript { lambda: BigNum::new(), x3: BigNum::new(), y3: BigNum::new() }
    }

    unconstrained fn from_j(j_tx: JTranscript<BigNumInstance, BigNum>, bn: BigNumInstance) -> Self {
        AffineTranscript::from_jacobian_transcript([j_tx].as_slice(), bn)[0]
    }

    unconstrained pub fn from_j_with_hint(
        j_tx: JTranscript<BigNumInstance, BigNum, CurveParams>,
        inverse: BigNum,
        bn: BigNumInstance
    ) -> AffineTranscript<BigNumInstance, BigNum> {
        let z_inv = inverse;
        let zz = bn.__mulmod(z_inv, z_inv);
        let zzz = bn.__mulmod(zz, z_inv);
        let lambda = bn.__mulmod(j_tx.lambda_numerator, z_inv);
        let x3 = bn.__mulmod(j_tx.x3, zz);
        let y3 = bn.__mulmod(j_tx.y3, zzz);
        AffineTranscript { x3, y3, lambda }
    }

    unconstrained fn from_jacobian_transcript(
        j_tx: [JTranscript<BigNumInstance, BigNum>],
        bn: BigNumInstance
    ) -> [AffineTranscript<BigNumInstance, BigNum>] {
        let mut result: [AffineTranscript<BigNumInstance, BigNum>] = &[];

        let mut inverses: [BigNum] = &[];
        for i in 0..j_tx.len() {
            inverses = inverses.push_back(j_tx[i].z3);
        }

        // tadaa
        let inverses = bn.__batch_invert(inverses.as_array());

        for i in 0..j_tx.len() {
            let z_inv = inverses[i];
            let zz = bn.__mulmod(z_inv, z_inv);
            let zzz = bn.__mulmod(zz, z_inv);
            let lambda = bn.__mulmod(j_tx[i].lambda_numerator, z_inv);
            let x3 = bn.__mulmod(j_tx[i].x3, zz);
            let y3 = bn.__mulmod(j_tx[i].y3, zzz);
            result = result.push_back(AffineTranscript { lambda, x3, y3 });
        }
        result
    }
}

/**
 * @brief A lookup table we use when performing scalar multiplications.
 * @description We slice scalar multiplier into 4 bit chunks represented
 * in windowed non-adjacent form ([-15, -13, ..., 15])
 * We compute a table of point multiples that map to the 4-bit WNAF values T = ([-15[P], -13[P], ..., 15[P]])
 * We set an accumulator to equal T[most significant WNAF slice]
 * We then iterate over our remaining bit slices (starting with most significant slice)
 * For each iteration `i` we double the accumulator 4 times and then add `T[slice[i]]` into the accumulator.
 * For small multiscalar multiplications (i.e. <512 points) this produces the minimal number of addition operations.
 **/
struct PointTable<BigNumInstance, BigNum> {
    x: [BigNum; 16],
    y: [BigNum; 16],
    z: [BigNum; 16],
    transcript: [JTranscript<BigNumInstance, BigNum>; 8]
}

impl<BigNumInstance, BigNum> PointTable<BigNumInstance, BigNum> where BigNum: BigNumTrait, BigNumInstance: BigNumInstanceTrait<BigNum> {
    fn empty() -> Self {
        PointTable {
            x: [BigNum::new(); 16],
            y: [BigNum::new(); 16],
            z: [BigNum::new(); 16],
            transcript: [JTranscript::new(); 8]
        }
    }
    /**
     * @brief make a new PointTable from an input point
     * @description we use "windowed non-adjacent form" representation 
     * to reduce the number of group operations required for the table
     * [-15P, -13P, ..., 15P] requires 8 group operations
     * [0, P, ..., 15P] requires 14 group operations.
     * group operations are expensive!
     **/
    unconstrained fn new<CurveParams>(
        bn: BigNumInstance,
        P: CurveJ<BigNumInstance, BigNum, CurveParams>
    ) -> Self where CurveParams: CurveParamsTrait<BigNumInstance, BigNum> {
        let mut result = PointTable {
            x: [BigNum::new(); 16],
            y: [BigNum::new(); 16],
            z: [BigNum::new(); 16],
            transcript: [JTranscript::new(); 8]
        };
        let op = P.dbl(bn);
        let D2 = op.0;
        result.transcript[0] = op.1;
        result.x[7] = P.x;
        result.y[7] = bn.__negate(P.y);
        result.z[7] = P.z;
        result.x[8] = P.x;
        result.y[8] = P.y;
        result.z[8] = P.z;
        let mut A = P;
        for i in 1..8 {
            let op = D2.incomplete_add(A, bn);
            A = op.0;
            result.transcript[i] = op.1;
            result.x[8+i] = A.x;
            result.y[8+i] = A.y;
            result.z[8+i] = A.z;
            result.x[7-i] = A.x;
            result.y[7-i] = bn.__negate(A.y);
            result.z[7-i] = A.z;
        }

        result
    }

    /**
     * @brief get a value out of the lookup table
     **/
    unconstrained fn get<CurveParams>(
        self,
        idx: u8
    ) -> CurveJ<BigNumInstance, BigNum, CurveParams> where CurveParams: CurveParamsTrait<BigNumInstance, BigNum> {
        CurveJ { x: self.x[idx], y: self.y[idx], z: self.z[idx], is_infinity: false }
    }
}

/**
 * @brief construct from BigCurve
 **/
impl<BigNumInstance, BigNum, CurveParams> std::convert::From<BigCurve<BigNumInstance, BigNum, CurveParams>> for CurveJ<BigNumInstance, BigNum, CurveParams> where BigNum: BigNumTrait, CurveParams: CurveParamsTrait<BigNumInstance, BigNum> {
    fn from(affine_point: BigCurve<BigNumInstance, BigNum, CurveParams>) -> Self {
        CurveJ{ x: affine_point.x, y: affine_point.y, z: BigNum::one(), is_infinity: affine_point.is_infinity }
    }
}

impl<BigNumInstance, BigNum, CurveParams> CurveJ<BigNumInstance, BigNum, CurveParams> where BigNum: BigNumTrait, BigNumInstance: BigNumInstanceTrait<BigNum>, CurveParams: CurveParamsTrait<BigNumInstance, BigNum> {
    /**
 * @brief are two Jacobian points equal?
 * @description only really used in tests for now.
 **/
    fn eq(bn: BigNumInstance, self, other: Self) -> bool {
        // if x == y then (X1 / Z1 * Z1 = X2 / Z2 * Z2)
        //            and (Y1 / Z1 * Z1 * Z1 = Y2 / Z2 * Z2 * Z2)

        // we can check this by validating that:
        // X1 * Z2 * Z2 == X2 * Z1 * Z1
        // Y1 * Z2 * Z2 * Z2 == Y2 * Z1 * Z1 * Z1
        let z1 = self.z;
        let z2 = other.z;
        let z1z1 = bn.__mulmod(z1, z1);
        let z1z1z1 = bn.__mulmod(z1z1, z1);
        let z2z2 = bn.__mulmod(z2, z2);
        let z2z2z2 = bn.__mulmod(z2z2, z2);

        let x_lhs = bn.__mulmod(self.x, z2z2);
        let x_rhs = bn.__mulmod(other.x, z1z1);
        let y_lhs = bn.__mulmod(self.y, z2z2z2);
        let y_rhs = bn.__mulmod(other.y, z1z1z1);
        let lhs_infinity = self.is_infinity;
        let rhs_infinity = other.is_infinity;
        let both_not_infinity = !(lhs_infinity & rhs_infinity);
        let both_infinity = !both_not_infinity;

        let mut points_equal = (bn.eq(x_lhs, x_rhs) & bn.eq(y_lhs, y_rhs)) & both_not_infinity;
        points_equal = points_equal | both_infinity;
        points_equal
    }
    /**
     * @brief negate a point
     **/
    fn neg(self, bn: BigNumInstance) -> Self {
        CurveJ { x: self.x, y: bn.__negate(self.y), z: self.z, is_infinity: self.is_infinity }
    }

    unconstrained fn new() -> Self {
        CurveJ { x: BigNum::new(), y: BigNum::new(), z: BigNum::new(), is_infinity: false }
    }

    unconstrained fn point_at_infinity() -> Self {
        CurveJ { x: BigNum::new(), y: BigNum::new(), z: BigNum::new(), is_infinity: true }
    }

    unconstrained fn sub(
        self,
        p2: Self,
        bn: BigNumInstance
    ) -> (Self, JTranscript<BigNumInstance, BigNum>) {
        self.add(p2.neg(bn), bn)
    }

    unconstrained fn add(
        self,
        p2: Self,
        bn: BigNumInstance
    ) -> (Self, JTranscript<BigNumInstance, BigNum>) {
        // TODO: once we have linear expressions as unconstrained fns, replace this with something that has no addmods, submods
        let X1 = self.x;
        let X2 = p2.x;
        let Y1 = self.y;
        let Y2 = p2.y;
        let Z1 = self.z;
        let Z2 = p2.z;
        let Z2Z2 = bn.__mulmod(Z2, Z2);
        let Z1Z1 = bn.__mulmod(Z1, Z1);
        let Z2Z2Z2 = bn.__mulmod(Z2Z2, Z2);
        let Z1Z1Z1 = bn.__mulmod(Z1Z1, Z1);
        let U1 = bn.__mulmod(X1, Z2Z2);
        let U2 = bn.__mulmod(X2, Z1Z1);
        let S1 = bn.__mulmod(Y1, Z2Z2Z2);
        let S2 = bn.__mulmod(Y2, Z1Z1Z1);
        let R = bn.__submod(S2, S1);

        // x1*z2*z2 == x2*z1*z1 => U2 == U2
        let x_equal_predicate = bn.eq(U2, U1);
        let y_equal_predicate = bn.eq(S2, S1);
        let lhs_infinity = self.is_infinity;
        let rhs_infinity = p2.is_infinity;
        let double_predicate = x_equal_predicate & y_equal_predicate & !lhs_infinity & !rhs_infinity;
        let add_predicate = !x_equal_predicate & !lhs_infinity & !rhs_infinity;
        let infinity_predicate = (x_equal_predicate & !y_equal_predicate) | (lhs_infinity & rhs_infinity);
        println(f"infinity predicate = {infinity_predicate}");
        println(f"double predicate = {double_predicate}");
        println(f"add predicate = {add_predicate}");
        println(f"lhs infinity = {lhs_infinity}");
        println(f"rhs infinity = {rhs_infinity}");
        let mut result: (Self, JTranscript<BigNumInstance, BigNum>) = (CurveJ::new(), JTranscript::new());
        if (double_predicate) {
            result = self.dbl(bn);
        } else if (add_predicate) {
            result = self.incomplete_add(p2, bn);
        } else if (infinity_predicate) {
            result = (CurveJ::point_at_infinity(), JTranscript::new() );
        } else if (lhs_infinity & !rhs_infinity) {
            result = (p2, JTranscript::new());
        } else if (rhs_infinity & !lhs_infinity) {
            result = (self, JTranscript::new());
        }
        result
        // let (_, PP): (BigNum, BigNum ) = BigNum::__compute_quadratic_expression([[U2, U1]], [[false, true]], [[U2, U1]], [[false, true]], [], []);
        // let (_, X3): (BigNum, BigNum ) = BigNum::__compute_quadratic_expression(
        //     [[BigNum::new(), PP], [R, BigNum::new()]],
        //     [[false, true], [false, false]],
        //     [[U1, U2], [R, BigNum::new()]],
        //     [[false, false], [false, false]],
        //     [],
        //     []
        // );
        // let (_, U1S2_minus_U2S1): (BigNum, BigNum ) = BigNum::__compute_quadratic_expression(
        //     [[U1], [U2]],
        //     [[false], [true]],
        //     [[S2], [S1]],
        //     [[false], [false]],
        //     [],
        //     []
        // );
        // let (_, Y3): (BigNum, BigNum ) = BigNum::__compute_quadratic_expression(
        //     [[PP], [X3]],
        //     [[false], [false]],
        //     [[U1S2_minus_U2S1], [R]],
        //     [[false], [true]],
        //     [],
        //     []
        // );
        // let Z1Z2 = Z1.__mulmod(Z2);
        // let (_, Z3): (BigNum, BigNum ) = BigNum::__compute_quadratic_expression(
        //     [[Z1Z2, BigNum::new()]],
        //     [[false, false]],
        //     [[U2, U1]],
        //     [[false, true]],
        //     [],
        //     []
        // );
        // (
        //     CurveJ { x: X3, y: Y3, z: Z3, is_infinity: false }, JTranscript { lambda_numerator: R, x3: X3, y3: Y3, z3: Z3 }
        // )
    }

    /**
     * @brief Add two points together.
     * @description Only uses incomplete formulae.
     * With our use of offset generators, we should never need to handle edge cases.
     * (when constraining operations, we simply assert the input x-coordinates are not equal)
     * @note This method minimizes the number of calls to `compute_quadratic_expression`,
     * which is NOT the same as minimizing the number of multiplications.
     **/
    unconstrained fn incomplete_add(self, p2: Self, bn: BigNumInstance) -> (Self, JTranscript<BigNumInstance, BigNum>) {
        // TODO: once we have linear expressions as unconstrained fns, replace this with something that has no addmods, submods
        let X1 = self.x;
        let X2 = p2.x;
        let Y1 = self.y;
        let Y2 = p2.y;
        let Z1 = self.z;
        let Z2 = p2.z;
        let Z2Z2 = bn.__mulmod(Z2, Z2);
        let Z1Z1 = bn.__mulmod(Z1, Z1);
        let Z2Z2Z2 = bn.__mulmod(Z2Z2, Z2);
        let Z1Z1Z1 = bn.__mulmod(Z1Z1, Z1);
        let U1 = bn.__mulmod(X1, Z2Z2);
        let U2 = bn.__mulmod(X2, Z1Z1);
        let S1 = bn.__mulmod(Y1, Z2Z2Z2);
        let S2 = bn.__mulmod(Y2, Z1Z1Z1);
        let R = bn.__submod(S2, S1);

        let (_, PP): (BigNum, BigNum ) = bn.__compute_quadratic_expression([[U2, U1]], [[false, true]], [[U2, U1]], [[false, true]], [], []);

        let (_, X3): (BigNum, BigNum ) = bn.__compute_quadratic_expression(
            [[BigNum::new(), PP], [R, BigNum::new()]],
            [[false, true], [false, false]],
            [[U1, U2], [R, BigNum::new()]],
            [[false, false], [false, false]],
            [],
            []
        );

        let (_, U1S2_minus_U2S1): (BigNum, BigNum ) = bn.__compute_quadratic_expression(
            [[U1], [U2]],
            [[false], [true]],
            [[S2], [S1]],
            [[false], [false]],
            [],
            []
        );
        let (_, Y3): (BigNum, BigNum ) = bn.__compute_quadratic_expression(
            [[PP], [X3]],
            [[false], [false]],
            [[U1S2_minus_U2S1], [R]],
            [[false], [true]],
            [],
            []
        );
        let Z1Z2 = bn.__mulmod(Z1, Z2);
        let (_, Z3): (BigNum, BigNum ) = bn.__compute_quadratic_expression(
            [[Z1Z2, BigNum::new()]],
            [[false, false]],
            [[U2, U1]],
            [[false, true]],
            [],
            []
        );

        (
            CurveJ { x: X3, y: Y3, z: Z3, is_infinity: false }, JTranscript { lambda_numerator: R, x3: X3, y3: Y3, z3: Z3 }
        )
    }

    /**
     * @brief Double a point
     * @note This method minimizes the number of calls to `compute_quadratic_expression`,
     * which is NOT the same as minimizing the number of multiplications.
     **/
    unconstrained fn dbl(self, bn: BigNumInstance) -> (Self, JTranscript<BigNumInstance, BigNum>) {
        let X1 = self.x;
        let Y1 = self.y;
        let Z1 = self.z;

        let (_, YY_mul_2): (BigNum, BigNum ) = bn.__compute_quadratic_expression([[Y1]], [[false]], [[Y1, Y1]], [[false, false]], [], []);
        let (_, XX_mul_3): (BigNum, BigNum ) = bn.__compute_quadratic_expression(
            [[X1]],
            [[false]],
            [[X1, X1, X1]],
            [[false, false, false]],
            [],
            []
        );
        let (_, D): (BigNum, BigNum ) = bn.__compute_quadratic_expression([[X1, X1]], [[false, false]], [[YY_mul_2]], [[false]], [], []);
        let mut (_, X3): (BigNum, BigNum ) = bn.__compute_quadratic_expression(
            [[XX_mul_3]],
            [[false]],
            [[XX_mul_3]],
            [[false]],
            [D, D],
            [true, true]
        );
        let (_, Y3): (BigNum, BigNum ) = bn.__compute_quadratic_expression(
            [[XX_mul_3], [YY_mul_2]],
            [[false], [true]],
            [[D, X3], [YY_mul_2, YY_mul_2]],
            [[false, true], [false, false]],
            [],
            []
        );
        // TODO: tidy this up and make more efficient?!
        if (CurveParams::a().get_limb(0) != 0) {
            let ZZ = bn.mul(Z1, Z1);
            let AZZZZ = bn.mul(bn.mul(ZZ, ZZ), (CurveParams::a()));
            X3 = bn.add(X3, AZZZZ);
        }
        let (_, Z3): (BigNum, BigNum ) = bn.__compute_quadratic_expression([[Y1]], [[false]], [[Z1, Z1]], [[false, false]], [], []);
        (
            CurveJ { x: X3, y: Y3, z: Z3, is_infinity: false }, JTranscript { lambda_numerator: XX_mul_3, x3: X3, y3: Y3, z3: Z3 }
        )
    }

    fn offset_generator() -> Self {
        let result = CurveParams::offset_generator();
        Self { x: result[0], y: result[1], z: BigNum::one(), is_infinity: false }
    }

    fn offset_generator_final() -> Self {
        let result = CurveParams::offset_generator_final();
        Self { x: result[0], y: result[1], z: BigNum::one(), is_infinity: false }
    }

    fn one() -> Self {
        let result = CurveParams::one();
        Self { x: result[0], y: result[1], z: BigNum::one(), is_infinity: false }
    }

    fn conditional_select(lhs: Self, rhs: Self, predicate: bool) -> Self {
        let mut result = rhs;
        if (predicate) {
            result = lhs;
        }
        result
    }

    /**
     * @brief Perform an ecc scalar multiplication and output the generated AffineTranscript
     **/
    unconstrained fn mul<let NScalarSlices: u32>(
        self,
        scalar: ScalarField<NScalarSlices>,
        bn: BigNumInstance
    ) -> (Self, [AffineTranscript<BigNumInstance, BigNum>]) {
        let mut transcript: [JTranscript<BigNumInstance, BigNum>] = &[];

        let input: Self = CurveJ::conditional_select(CurveJ::one(), self, self.is_infinity);
        let scalar: ScalarField<NScalarSlices> = ScalarField::conditional_select(ScalarField::zero(), scalar, self.is_infinity);

        let T = PointTable::new(bn, input);
        for i in 0..8 {
            transcript = transcript.push_back(T.transcript[i]);
        }

        let mut accumulator: Self = CurveJ::offset_generator();
        let op = accumulator.incomplete_add(T.get(scalar.base4_slices[0]), bn);
        transcript = transcript.push_back(op.1);
        accumulator = op.0;
        for i in 1..NScalarSlices {
            for _ in 0..4 {
                let op = accumulator.dbl(bn);
                accumulator = op.0;
                transcript = transcript.push_back(op.1);
            }
            let op = accumulator.incomplete_add(T.get(scalar.base4_slices[i]), bn);
            transcript = transcript.push_back(op.1);
            accumulator = op.0;
        }

        if scalar.skew {
            let op = accumulator.incomplete_add(input.neg(bn), bn);
            transcript = transcript.push_back(op.1);
            accumulator = op.0;
        } else {
            transcript = transcript.push_back(JTranscript::new());
        }

        let op = accumulator.sub(CurveJ::offset_generator_final(), bn);
        transcript = transcript.push_back(op.1);
        accumulator = op.0;

        println(f"FINAL ACCUMULATOR = {accumulator}");
        let affine_transcript: [AffineTranscript<BigNumInstance, BigNum>] = AffineTranscript::from_jacobian_transcript(transcript, bn);

        (accumulator, affine_transcript)
    }

    /**
     * @brief Perform an ecc scalar multiplication and output the generated AffineTranscript
     **/
    unconstrained fn msm_partial<let Size: u32, let NScalarSlices: u32>(
        mut points: [Self; Size],
        mut scalars: [ScalarField<NScalarSlices>; Size],
        bn: BigNumInstance
    ) -> (Self, [JTranscript<BigNumInstance, BigNum>]) {
        let mut transcript: [JTranscript<BigNumInstance, BigNum>] = &[];
        let mut tables: [PointTable<BigNumInstance, BigNum>; Size] = [PointTable::empty(); Size];

        let mut _inputs: [Self; Size] = [CurveJ::new(); Size];
        let mut  _scalars: [ScalarField<NScalarSlices>; Size] = [ScalarField::new(); Size];
        for i in 0..Size {
            _inputs[i] = CurveJ::conditional_select(CurveJ::one(), points[i], points[i].is_infinity);
            _scalars[i] = ScalarField::conditional_select(ScalarField::zero(), scalars[i], points[i].is_infinity);
        }
        points = _inputs;
        scalars = _scalars;

        for i in 0..Size {
            tables[i] = PointTable::new(bn, points[i]);
            for j in 0..8 {
                transcript = transcript.push_back(tables[i].transcript[j]);
            }
        }

        let mut accumulator: Self = CurveJ::offset_generator();
        let op = accumulator.incomplete_add(tables[0].get(scalars[0].base4_slices[0]), bn);
        transcript = transcript.push_back(op.1);
        accumulator = op.0;

        for i in 1..Size {
            println(f"SizeIt = {i}");
            let op = accumulator.incomplete_add(tables[i].get(scalars[i].base4_slices[0]), bn);
            transcript = transcript.push_back(op.1);
            accumulator = op.0;
        }
        for i in 1..NScalarSlices {
            println(f"ScalarSliceIt = {i}");
            for _ in 0..4 {
                let op = accumulator.dbl(bn);
                accumulator = op.0;
                transcript = transcript.push_back(op.1);
            }
            for j in 0..Size {
                let op = accumulator.incomplete_add(tables[j].get(scalars[j].base4_slices[i]), bn);
                transcript = transcript.push_back(op.1);
                accumulator = op.0;
            }
        }

        for i in 0..Size {
            if scalars[i].skew {
                let op = accumulator.incomplete_add(points[i].neg(bn), bn);
                transcript = transcript.push_back(op.1);
                accumulator = op.0;
            } else {
                transcript = transcript.push_back(JTranscript::new());
            }
        }
        // let op = accumulator.sub(CurveJ::offset_generator_final());
        // transcript = transcript.push_back(op.1);
        // accumulator = op.0;
        // let affine_transcript: [AffineTranscript<BigNum>] = AffineTranscript::from_jacobian_transcript(transcript);

        (accumulator, transcript)
    }

    /**
     * @brief Perform an ecc scalar multiplication and output the generated AffineTranscript
     **/
    unconstrained fn msm<let Size: u32, let NScalarSlices: u32>(
        mut points: [Self; Size],
        mut scalars: [ScalarField<NScalarSlices>; Size],
        bn: BigNumInstance
    ) -> (Self, [AffineTranscript<BigNumInstance, BigNum>]) {
        let mut (accumulator, transcript) = CurveJ::msm_partial(points, scalars, bn);
        let op = accumulator.sub(CurveJ::offset_generator_final(), bn);
        transcript = transcript.push_back(op.1);
        accumulator = op.0;
        let affine_transcript: [AffineTranscript<BigNumInstance, BigNum>] = AffineTranscript::from_jacobian_transcript(transcript, bn);

        (accumulator, affine_transcript)
    }

    unconstrained fn compute_linear_expression_transcript<let NScalarSlices: u64, let NMuls: u64, let NAdds: u64>(
        mul_points: [BigCurve<BigNumInstance, BigNum, CurveParams>; NMuls],
        mul_scalars: [ScalarField<NScalarSlices>; NMuls],
        add_points: [BigCurve<BigNumInstance, BigNum, CurveParams>; NAdds],
        bn: BigNumInstance
    ) -> AffineLinearExpressionTranscript<BigNumInstance, BigNum, CurveParams, NScalarSlices, NMuls, NAdds> {
        let mut mul_j: [CurveJ<BigNumInstance, BigNum, CurveParams>; NMuls] = [CurveJ::new(); NMuls];
        let mut add_j: [CurveJ<BigNumInstance, BigNum, CurveParams>; NAdds] = [CurveJ::new(); NAdds];
        for i in 0..NMuls {
            mul_j[i] = CurveJ::from(mul_points[i]);
        }
        for i in 0..NAdds {
            add_j[i] = CurveJ::from(add_points[i]);
        }

        let mut jacobian_transcript: LinearExpressionTranscript<BigNumInstance, BigNum, CurveParams,NScalarSlices,NMuls,NAdds> = LinearExpressionTranscript {
            table_transcript: [[JTranscript::new(); 8]; NMuls],
            msm_double_transcript: [[JTranscript::new(); 4]; NScalarSlices],
            msm_add_transcript: [[JTranscript::new(); NMuls]; NScalarSlices],
            skew_transcript: [JTranscript::new(); NMuls],
            add_transcript: [JTranscript::new(); NAdds],
            offset_generator_transcript: JTranscript::new()
        };

        // #####
        let mut tables: [PointTable<BigNumInstance, BigNum>; NMuls] = [PointTable::empty(); NMuls];

        let mut _inputs: [Self; NMuls] = [CurveJ::new(); NMuls];
        let mut  _scalars: [ScalarField<NScalarSlices>; NMuls] = [ScalarField::new(); NMuls];
        for i in 0..NMuls {
            _inputs[i] = CurveJ::conditional_select(CurveJ::one(), mul_j[i], mul_j[i].is_infinity);
            _scalars[i] = ScalarField::conditional_select(ScalarField::zero(), mul_scalars[i], mul_j[i].is_infinity);
        }
        let points = _inputs;
        let scalars = _scalars;

        for i in 0..NMuls {
            tables[i] = PointTable::new(bn, points[i]);
            for j in 0..8 {
                jacobian_transcript.table_transcript[i][j] = tables[i].transcript[j];
            }
        }

        let mut accumulator: Self = CurveJ::offset_generator();
        let op = accumulator.incomplete_add(tables[0].get(scalars[0].base4_slices[0]), bn);
        jacobian_transcript.msm_add_transcript[0][0] = op.1;
        accumulator = op.0;

        for i in 1..NMuls {
            println(f"SizeIt = {i}");
            let op = accumulator.incomplete_add(tables[i].get(scalars[i].base4_slices[0]), bn);
            jacobian_transcript.msm_add_transcript[0][i] = op.1;
            accumulator = op.0;
        }
        for i in 1..NScalarSlices {
            println(f"ScalarSliceIt = {i}");
            for j in 0..4 {
                let op = accumulator.dbl(bn);
                accumulator = op.0;
                jacobian_transcript.msm_double_transcript[i][j] = op.1;
            }
            for j in 0..NMuls {
                let op = accumulator.incomplete_add(tables[j].get(scalars[j].base4_slices[i]), bn);
                jacobian_transcript.msm_add_transcript[i][j] = op.1;
                accumulator = op.0;
            }
        }

        for i in 0..NMuls {
            if scalars[i].skew {
                let op = accumulator.incomplete_add(points[i].neg(bn), bn);
                jacobian_transcript.skew_transcript[i] = op.1;
                accumulator = op.0;
            } else {
                jacobian_transcript.skew_transcript[i] = op.1;
            }
        }

        for i in 0..NAdds {
            let op = accumulator.incomplete_add(add_j[i], bn);
            jacobian_transcript.add_transcript[i] = op.1;
            accumulator = op.0;
        }

        let op = accumulator.sub(CurveJ::offset_generator_final(), bn);
        jacobian_transcript.offset_generator_transcript = op.1;
        accumulator = op.0;

        let affine_transcript: AffineLinearExpressionTranscript<BigNumInstance, BigNum, CurveParams, NScalarSlices, NMuls, NAdds> = AffineLinearExpressionTranscript::from_jtranscript(jacobian_transcript, bn);
        affine_transcript
    }
}

struct LinearExpressionTranscript<BigNumInstance, BigNum, CurveParams, let NScalarSlices: u64, let NMuls: u64, let NAdds: u64>{
    table_transcript: [[JTranscript<BigNumInstance, BigNum, CurveParams>; 8]; NMuls],
    msm_double_transcript: [[JTranscript<BigNumInstance, BigNum, CurveParams>; 4]; NScalarSlices],
    msm_add_transcript: [[JTranscript<BigNumInstance, BigNum, CurveParams>; NMuls]; NScalarSlices],
    skew_transcript: [JTranscript<BigNumInstance, BigNum, CurveParams>; NMuls],
    add_transcript: [JTranscript<BigNumInstance, BigNum, CurveParams>; NAdds],
    offset_generator_transcript: JTranscript<BigNumInstance, BigNum, CurveParams>
}

struct AffineLinearExpressionTranscript<BigNumInstance, BigNum, CurveParams, let NScalarSlices: u64, let NMuls: u64, let NAdds: u64>{
    table_transcript: [[AffineTranscript<BigNumInstance, BigNum, CurveParams>; 8]; NMuls],
    msm_double_transcript: [[AffineTranscript<BigNumInstance, BigNum, CurveParams>; 4]; NScalarSlices],
    msm_add_transcript: [[AffineTranscript<BigNumInstance, BigNum, CurveParams>; NMuls]; NScalarSlices],
    skew_transcript: [AffineTranscript<BigNumInstance, BigNum, CurveParams>; NMuls],
    add_transcript: [AffineTranscript<BigNumInstance, BigNum, CurveParams>; NAdds],
    offset_generator_transcript: AffineTranscript<BigNumInstance, BigNum, CurveParams>
}

// unconstrained fn from_jacobian_transcript(j_tx: [JTranscript<BigNum>]) -> [AffineTranscript<BigNum>] {
//     let mut result: [AffineTranscript<BigNum>] = &[];

//     let mut inverses: [BigNum] = &[];
//     for i in 0..j_tx.len() {
//         inverses = inverses.push_back(j_tx[i].z3);
//     }

//     // tadaa
//     BigNum::__batch_invert(&mut inverses);

//     for i in 0..j_tx.len() {
//         let z_inv = inverses[i];
//         let zz = z_inv.__mulmod(z_inv);
//         let zzz = zz.__mulmod(z_inv);
//         let lambda = j_tx[i].lambda_numerator.__mulmod(z_inv);
//         let x3 = j_tx[i].x3.__mulmod(zz);
//         let y3 = j_tx[i].y3.__mulmod(zzz);
//         result = result.push_back(AffineTranscript { lambda, x3, y3 });
//     }
//     result
// }

impl<BigNumInstance, BigNum, CurveParams, let NScalarSlices: u64, let NMuls: u64, let NAdds: u64>  AffineLinearExpressionTranscript<BigNumInstance, BigNum, CurveParams,NScalarSlices,NMuls,NAdds> where BigNum: BigNumTrait, BigNumInstance: BigNumInstanceTrait<BigNum>, CurveParams: CurveParamsTrait<BigNumInstance, BigNum> {

    unconstrained fn from_jtranscript(
        jtranscript: LinearExpressionTranscript<BigNumInstance, BigNum, CurveParams,  NScalarSlices,  NMuls,  NAdds>,
        bn: BigNumInstance
    ) -> Self {
        let mut inverses: [BigNum] = &[];

        for i in 0..NMuls {
            for j in 0..8 {
                inverses = inverses.push_back(jtranscript.table_transcript[i][j].z3);
            }
        }
        for i in 0..NMuls {
            inverses = inverses.push_back(jtranscript.msm_add_transcript[0][i].z3);
        }
        for i in 1..NScalarSlices {
            for j in 0..4 {
                inverses = inverses.push_back(jtranscript.msm_double_transcript[i][j].z3);
            }

            for j in 0..NMuls {
                inverses = inverses.push_back(jtranscript.msm_add_transcript[i][j].z3);
            }
        }

        for i in 0..NMuls {
            inverses = inverses.push_back(jtranscript.skew_transcript[i].z3);
        }
        for i in 0..NAdds {
            inverses = inverses.push_back(jtranscript.add_transcript[i].z3);
        }
        inverses = inverses.push_back(jtranscript.offset_generator_transcript.z3);

        let inverses = bn.__batch_invert(inverses.as_array());

        let mut result: AffineLinearExpressionTranscript<BigNumInstance, BigNum, CurveParams,NScalarSlices,NMuls,NAdds> = AffineLinearExpressionTranscript {
            table_transcript: [[AffineTranscript::new(); 8]; NMuls],
            msm_double_transcript: [[AffineTranscript::new(); 4]; NScalarSlices],
            msm_add_transcript: [[AffineTranscript::new(); NMuls]; NScalarSlices],
            skew_transcript: [AffineTranscript::new(); NMuls],
            add_transcript: [AffineTranscript::new(); NAdds],
            offset_generator_transcript: AffineTranscript::new()
        };

        let mut it = 0;
        for i in 0..NMuls {
            for j in 0..8 {
                result.table_transcript[i][j] = AffineTranscript::from_j_with_hint(jtranscript.table_transcript[i][j], inverses[it], bn);
                it+=1;
            }
        }
        for i in 0..NMuls {
            result.msm_add_transcript[0][i] = AffineTranscript::from_j_with_hint(jtranscript.msm_add_transcript[0][i], inverses[it], bn);
            it+=1;
        }
        for i in 1..NScalarSlices {
            for j in 0..4 {
                result.msm_double_transcript[i][j] = AffineTranscript::from_j_with_hint(jtranscript.msm_double_transcript[i][j], inverses[it], bn);
                it+=1;
            }

            for j in 0..NMuls {
                result.msm_add_transcript[i][j] = AffineTranscript::from_j_with_hint(jtranscript.msm_add_transcript[i][j], inverses[it], bn);
                it+=1;
            }
        }

        for i in 0..NMuls {
            result.skew_transcript[i] = AffineTranscript::from_j_with_hint(jtranscript.skew_transcript[i], inverses[it], bn);
            it+=1;
        }
        for i in 0..NAdds {
            result.add_transcript[i] = AffineTranscript::from_j_with_hint(jtranscript.add_transcript[i], inverses[it], bn);
            it+=1;
        }

        result.offset_generator_transcript = AffineTranscript::from_j_with_hint(jtranscript.offset_generator_transcript, inverses[it], bn);

        result
    }
}
impl<BigNumInstance, BigNum, CurveParams, let NBits: u64, let NScalarSlices: u64, let NMuls: u64, let NAdds: u64> LinearExpressionTranscript<BigNumInstance, BigNum, CurveParams, NBits,  NScalarSlices,  NMuls,  NAdds> where BigNum: BigNumTrait, BigNumInstance: BigNumInstanceTrait<BigNum>, CurveParams: CurveParamsTrait<BigNumInstance, BigNum> {}

// TODO: this should probably be in bignum
// unconstrained pub fn batch_invert<BigNum, BigNumInstance>(
//     x: &mut [BigNum],
//     bn: BigNumInstance
// ) where BigNum: BigNumTrait, BigNumInstance: BigNumInstanceTrait<BigNum> {
//     let mut accumulator: BigNum = BigNum::one();

//     let mut temporaries: [BigNum] = &[];
//     for i in 0..x.len() {
//         temporaries = temporaries.push_back(accumulator);
//         if (x[i].__is_zero() == false) {
//             accumulator = bn.__mulmod(accumulator, x[i]);
//         }
//     }

//     accumulator = bn.__invmod(accumulator);
//     let mut T0: BigNum = BigNum::new();
//     for i in 0..x.len() {
//         let idx = x.len() - 1 - i;
//         if (x[idx].__is_zero() == false) {
//             T0 = bn.__mulmod(accumulator, temporaries[idx]);
//             accumulator = bn.__mulmod(accumulator, x[idx]);
//             x[idx] = T0;
//         }
//     }
// }
