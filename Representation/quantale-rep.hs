-- This script demonstrates quantale-based genetic operators as described
-- in the "MORK-MOSES Implementation Ideas" paper.

import Data.List (intercalate, (\\))

-- We'll start by defining the abstract structure of a commutative quantale.
-- This typeclass captures the core operations needed for variation.
class Quantale q where
  qjoin     :: q -> q -> q  -- Join operation (⊕), used for one type of crossover
  qtimes    :: q -> q -> q  -- Product operation (⊗), used for another crossover type
  qresiduum :: q -> q -> q  -- Residuum (→), used to calculate complements for masks
  qunit     :: q            -- The unit element (e) for the product operation

-- For our toy problem, we'll implement the simple probabilistic quantale
-- described in Section 9.1 of the paper.
-- The elements are values in [0,1]. We wrap Double in a newtype for clarity.
newtype ProbQuantale = Prob Double deriving (Show, Eq)

newtype LogicalQuantale = Logical Bool deriving (Show, Eq)

newtype StringQuantale = StrQnt String deriving (Show, Eq)

instance Quantale LogicalQuantale where
    qjoin (Logical a) (Logical b) = Logical (a || b) -- Logical OR
    qtimes (Logical a) (Logical b) = Logical (a && b) -- Logical AND
    qresiduum (Logical a) (Logical b) = Logical (not a || b) -- Implication
    qunit = Logical True -- Unit for AND is True

instance Quantale StringQuantale where
    qjoin (StrQnt a) (StrQnt b) = StrQnt (a ++ b) -- Concatenation
    qtimes (StrQnt a) (StrQnt b) = StrQnt (interleave a b)
        where
            interleave [] ys = ys
            interleave xs [] = xs
            interleave (x:xs) (y:ys) = x : y : interleave xs ys
    qresiduum (StrQnt a) (StrQnt b) = StrQnt (a \\ b)
    qunit = StrQnt "" -- Unit for concatenation is the empty string


instance Quantale ProbQuantale where
  -- Join (⊕) is defined as max.
  qjoin (Prob a) (Prob b) = Prob (max a b)

  -- Product (⊗) is defined as multiplication.
  qtimes (Prob a) (Prob b) = Prob (a * b)

  -- The residuum (a → b) is defined as min(1, b/a). 
  -- We handle the case where 'a' is 0 to avoid division by zero.
  qresiduum (Prob a) (Prob b) = if a == 0.0 then Prob 1.0 else Prob (min 1.0 (b / a))

  -- The unit (e) is 1.
  qunit = Prob 1.0

-- According to the paper, a program is a function mapping from some
-- context `x` to a quantale value `q`.
type Program x q = x -> q

-- Now, we implement the generic variation operators from Section 7.
-- These functions will work for any type that is an instance of our Quantale class.

-- 7.1 Crossover as Join and Product
joinCrossover :: (Quantale q) => Program x q -> Program x q -> Program x q
joinCrossover m1 m2 = \x -> qjoin (m1 x) (m2 x) -- m_child(x) = m1(x) ⊕ m2(x)

productCrossover :: (Quantale q) => Program x q -> Program x q -> Program x q
productCrossover m1 m2 = \x -> qtimes (m1 x) (m2 x) -- m_child(x) = m1(x) ⊗ m2(x)

-- 7.2 Mask-based Crossover
maskedCrossover :: (Quantale q) => Program x q -> Program x q -> Program x q -> Program x q
maskedCrossover mask m1 m2 = \x ->
  let p = mask x
      p_complement = qresiduum p qunit -- The complement p̄ is defined as e → p
      part1 = qtimes p (m1 x)
      part2 = qtimes p_complement (m2 x)
       in qjoin part1 part2 -- m_child(x) = (p(x) ⊗ m1(x)) ⊕ (p̄(x) ⊗ m2(x))

-- 7.3 Mutation as Perturbation
additiveMutation :: (Quantale q) => Program x q -> Program x q -> Program x q
additiveMutation m delta = \x -> qjoin (m x) (delta x) -- m+(x) = m(x) ⊕ δ(x) 

multiplicativeMutation :: (Quantale q) => Program x q -> Program x q -> Program x q
multiplicativeMutation m delta = \x -> qtimes (m x) (delta x) -- m×(x) = m(x) ⊗ δ(x)

data SensorData = SensorData {
  temperature :: Double,
  pressure    :: Double
} deriving (Show)

stringExample :: IO ()
stringExample = do
    putStrLn "\n\n--- String Quantale Example ---"
    
    -- Define two parent programs that operate on strings.
    let parentA :: Program String StringQuantale
        parentA s = StrQnt (take 3 s) -- First 3 characters
    
    let parentB :: Program String StringQuantale
        parentB s = StrQnt (reverse (take 3 (reverse s))) -- Last 3 characters
    
    -- Create a child using the joinCrossover function.
    let child = joinCrossover parentA parentB
    let prodChild = productCrossover parentA parentB
    let resChild = maskedCrossover parentA parentB parentB
    
    -- Evaluate the parents and child on a sample string.
    let sampleInput = "Haskell"
    putStrLn $ "Sample Input: " ++ sampleInput
    putStrLn $ "Parent A Output: " ++ show (parentA sampleInput)
    putStrLn $ "Parent B Output: " ++ show (parentB sampleInput)
    putStrLn $ "Child Output (Join Crossover): " ++ show (child sampleInput)
    putStrLn $ "Child Output (product Crossover): " ++ show (prodChild sampleInput)
    putStrLn $ "Child Output (res Crossover): " ++ show (resChild sampleInput)


    -- Expected: StrQnt "Haskell" (interleaving "Has" and "ell")

logicExample :: IO ()
logicExample = do
  putStrLn "\n\n--- Logic Quantale Example with Sensor Data ---"

  -- Define two parent programs that operate on SensorData.
  -- They return a boolean LogicQuantale value.
  let parentA :: Program SensorData LogicalQuantale
      parentA sensors = Logical (temperature sensors > 30.0) -- Is it hot?

  let parentB :: Program SensorData LogicalQuantale
      parentB sensors = Logical (pressure sensors < 1000.0) -- Is pressure low?

  -- Create a child using the SAME joinCrossover function as before.
  let child = joinCrossover parentA parentB

  -- Define some sample sensor readings.
  let highTemp = SensorData { temperature = 35.0, pressure = 1010.0 }
  let lowPressure = SensorData { temperature = 25.0, pressure = 990.0 }
  let normal = SensorData { temperature = 25.0, pressure = 1010.0 }

  -- Evaluate the parents and child on the sample data.
  putStrLn $ "High Temp Reading: " ++ show highTemp
  putStrLn $ "  - Parent A (Hot?): " ++ show (parentA highTemp)
  putStrLn $ "  - Parent B (Low Pressure?): " ++ show (parentB highTemp)
  putStrLn $ "  - Child (Hot OR Low Pressure?): " ++ show (child highTemp) -- Expected: Logic True

  putStrLn $ "\nLow Pressure Reading: " ++ show lowPressure
  putStrLn $ "  - Parent A (Hot?): " ++ show (parentA lowPressure)
  putStrLn $ "  - Parent B (Low Pressure?): " ++ show (parentB lowPressure)
  putStrLn $ "  - Child (Hot OR Low Pressure?): " ++ show (child lowPressure) -- Expected: Logic True

  putStrLn $ "\nNormal Reading: " ++ show normal
  putStrLn $ "  - Parent A (Hot?): " ++ show (parentA normal)
  putStrLn $ "  - Parent B (Low Pressure?): " ++ show (parentB normal)
  putStrLn $ "  - Child (Hot OR Low Pressure?): " ++ show (child normal) -- Expected: Logic False

-- Let's define a toy problem to see the operators in action.
-- Our program context 'x' will just be an Integer.
main :: IO ()
main = do
  -- Define two simple "parent" programs. They map an integer to a value in [0,1].
--   let parent1 :: Program Int ProbQuantale
--       parent1 x = Prob (abs (sin (fromIntegral x * 0.5)))

--   let parent2 :: Program Int ProbQuantale
--       parent2 x = Prob (abs (cos (fromIntegral x * 0.5)))

-- --   let logicalParent1 = 
--   -- Define a mask for maskedCrossover.
--   -- It picks from parent1 on even numbers and parent2 on odd numbers.
--   let mask :: Program Int ProbQuantale
--       mask x = if even x then qunit else Prob 0.0

--   -- Define a small random perturbation for mutation.
--   let perturbation :: Program Int ProbQuantale
--       perturbation _ = Prob 0.1

--   -- Generate children using the different operators.
--   let childJoin = joinCrossover parent1 parent2
--   let childProduct = productCrossover parent1 parent2
--   let childMasked = maskedCrossover mask parent1 parent2
--   let childMutated = additiveMutation parent1 perturbation

--   -- Print the results for a few inputs to see the effect.
--   putStrLn "--- Quantale Genetic Operators Toy Problem ---"
--   let inputs = [0..5]
--   putStrLn $ "Inputs: " ++ show inputs
--   putStrLn "\n--- Parent Programs ---"
--   putStrLn $ "Parent 1 outputs: " ++ show (map ((\(Prob v) -> v) . parent1) inputs)
--   putStrLn $ "Parent 2 outputs: " ++ show (map ((\(Prob v) -> v) . parent2) inputs)

--   putStrLn "\n--- Child Programs (Crossover) ---"
--   putStrLn $ "Join Crossover:     " ++ show (map ((\(Prob v) -> v) . childJoin) inputs)
--   putStrLn $ "Product Crossover:  " ++ show (map ((\(Prob v) -> v) . childProduct) inputs)
--   putStrLn $ "Masked Crossover:   " ++ show (map ((\(Prob v) -> v) . childMasked) inputs)

--   putStrLn "\n--- Child Program (Mutation) ---"
--   putStrLn $ "Additive Mutation:  " ++ show (map ((\(Prob v) -> v) . childMutated) inputs)

  logicExample
    -- stringExample