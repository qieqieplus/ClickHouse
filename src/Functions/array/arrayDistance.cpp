//
// Created by qieqie on 2021/9/26.
//
#include <Eigen/Core>

#include <Columns/ColumnTuple.h>
#include <DataTypes/DataTypeTuple.h>
#include <DataTypes/DataTypesNumber.h>
#include <Functions/FunctionFactory.h>
#include <Functions/FunctionHelpers.h>

#include "FunctionArrayMapped.h"

namespace DB
{
template <typename RT>
class FunctionArrayDistance : public IFunction
{
public:
    static constexpr auto name = "arrayDistance";
    static FunctionPtr create(ContextPtr) { return std::make_shared<FunctionArrayDistance<RT>>(); }
    String getName() const override { return name; }
    bool isVariadic() const override { return true; }
    //bool isSuitableForShortCircuitArgumentsExecution(const DataTypesWithConstInfo & /*arguments*/) const override { return true; }
    size_t getNumberOfArguments() const override { return 0; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        if (arguments.size() != 2)
        {
            throw Exception(
                "Function arrayDistance needs exactly two argument; passed " + toString(arguments.size()) + ".",
                ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);
        }

        size_t total_size = 1;
        DataTypePtr nested_type = nullptr;

        for (const auto & argument : arguments)
        {
            const DataTypeTuple * tuple_type = checkAndGetDataType<DataTypeTuple>(argument.type.get());
            if (!tuple_type)
            {
                throw Exception(
                    "Arguments of function arrayDistance must be tuple of array. Found " + argument.type->getName() + " instead.",
                    ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
            }

            const DataTypes & elements = tuple_type->getElements();
            if (elements.empty())
            {
                throw Exception("Nested array of function arrayDistance tuples can not be empty.", ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
            }
            for (const auto & element : elements)
            {
                const auto & type_ptr = getArrayNestedType(element);
                if (!nested_type)
                {
                    nested_type = type_ptr;
                }
                else if (!nested_type->equals(*type_ptr))
                {
                    throw Exception(
                        "Nested array of function arrayDistance tuples must have identical type.", ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
                }
            }

            total_size *= elements.size();
        }

        switch (nested_type->getTypeId())
        {
            case TypeIndex::UInt8:
                break;
            case TypeIndex::Float32:
                break;
            default:
                throw Exception(
                    "Nested type of function arrayDistance tuples must be array of uint8 or float32.",
                    ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
        }

        // return type tuple(float32)
        DataTypes types(total_size, nested_result_type);
        return std::make_shared<DataTypeTuple>(types);
    }

    ColumnPtr
    executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr & result_type, size_t /*input_rows_count*/) const override
    {
        size_t num_arguments = arguments.size();
        if (num_arguments != 2)
        {
            throw Exception("Function arrayDistance requires two arguments.", ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);
        }

        const auto & return_type = result_type;
        auto res_ptr = return_type->createColumn();
        ColumnTuple & res = assert_cast<ColumnTuple &>(*res_ptr);

        const DataTypeTuple * tuple_type = checkAndGetDataType<DataTypeTuple>(arguments[0].type.get());
        const DataTypes & elements = tuple_type->getElements();
        const DataTypePtr & nested_type = getArrayNestedType(elements.front());

        Columns first = getColumnsFromTuple(arguments[0].column->convertToFullColumnIfConst());
        Columns second = getConstColumnsFromTuple(arguments[1].column);

        switch (nested_type->getTypeId())
        {
            case TypeIndex::UInt8:
                executeMatrix<UInt8>(first, second, res);
                break;
            case TypeIndex::Float32:
                executeMatrix<Float32>(first, second, res);
                break;
            default:
                throw Exception(
                    "Nested array type " + nested_type->getName() + " is not supported as argument of function arrayDistance.",
                    ErrorCodes::ILLEGAL_COLUMN);
        }

        return res_ptr;
    }

private:
    static const DataTypePtr & getArrayNestedType(const DataTypePtr & type_ptr)
    {
        const DataTypeArray * array_type = checkAndGetDataType<DataTypeArray>(type_ptr.get());
        if (!array_type)
        {
            throw Exception(
                "Nested type of function arrayDistance tuples must be array of uint8 or float32. Found " + type_ptr->getName()
                    + " instead.",
                ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
        }
        return array_type->getNestedType();
    }

    static Columns getColumnsFromTuple(const ColumnPtr & column_ptr)
    {
        const ColumnTuple * tuple_column = checkAndGetColumn<ColumnTuple>(column_ptr.get());

        if (!tuple_column)
        {
            throw Exception(
                ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                "First argument of function arrayDistance should be tuple, got {}",
                column_ptr->getName());
        }
        return tuple_column->getColumnsCopy();
    }

    static Columns getConstColumnsFromTuple(const ColumnPtr & column_ptr)
    {
        const ColumnTuple * tuple_column = checkAndGetColumnConstData<ColumnTuple>(column_ptr.get());

        if (!tuple_column)
        {
            throw Exception(
                ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                "Second argument of function arrayDistance should be const tuple, got {}",
                column_ptr->getName());
        }
        return tuple_column->getColumnsCopy();
    }

    template <typename T>
    static bool executeMatrix(const Columns & first, const Columns & second, ColumnTuple & result)
    {
        std::vector<size_t> off_x, off_y;
        Eigen::MatrixX<RT> mx, my;

        if (!columnsToMatrix<T>(first, off_x, mx))
        {
            throw Exception("First argument of function arrayDistance must be array. ", ErrorCodes::ILLEGAL_COLUMN);
        }
        //std::cout << "matrix x:\n" << mx << std::endl;

        if (!columnsToMatrix<T>(second, off_y, my))
        {
            throw Exception("Arguments of function arrayDistance must be const array. ", ErrorCodes::ILLEGAL_COLUMN);
        }
        //std::cout << "matrix y:\n" << my << std::endl;

        if (mx.rows() && my.rows() && mx.rows() != my.rows())
        {
            throw Exception("Arrays of function arrayDistance have different sizes.", ErrorCodes::SIZES_OF_ARRAYS_DOESNT_MATCH);
        }

        size_t prev_x = 0, prev_y = 0;
        for (auto col : my.colwise())
        {
            auto norms = (mx.colwise() - col).colwise().norm();

            prev_x = 0;
            for (size_t i = 0; i < off_x.size(); ++i)
            {
                auto & column = result.getColumn(prev_y + i);
                for (size_t j = prev_x; j < off_x[i]; ++j)
                    column.insert(norms[j]);
                prev_x = off_x[i];
            }
            prev_y += off_x.size();
        }

        return true;
    }

    template <typename T>
    static bool columnsToMatrix(const Columns & columns, std::vector<size_t> & off, Eigen::MatrixX<RT> & mat)
    {
        const ColumnPtr & first_column = columns.front();
        const ColumnArray * first_array = checkAndGetColumn<ColumnArray>(first_column.get());
        if (!first_array)
            return false;

        const ColumnArray::Offsets & offsets = first_array->getOffsets();
        mat.resize(offsets.front(), offsets.size() * columns.size());

        ColumnArray::Offset col = 0;
        for (const ColumnPtr & column : columns)
        {
            const ColumnArray * column_array = checkAndGetColumn<ColumnArray>(column.get());
            if (!column_array)
                return false;

            const ColumnVector<T> * column_vec = checkAndGetColumn<ColumnVector<T>>(column_array->getData());
            const PaddedPODArray<T> & column_data = column_vec->getData();

            ColumnArray::Offset prev = 0;
            for (auto offset : offsets)
            {
                for (ColumnArray::Offset idx = 0; idx < offset - prev; ++idx)
                {
                    mat(idx, col) = column_data[prev + idx];
                }
                ++col;
                prev = offset;
            }
            off.emplace_back(col);
        }

        return true;
    }

    const DataTypePtr nested_result_type = std::make_shared<DataTypeNumber<RT>>();
};

void registerFunctionArrayDistance(FunctionFactory & factory)
{
    factory.registerFunction<FunctionArrayDistance<Float32>>();
}

}
