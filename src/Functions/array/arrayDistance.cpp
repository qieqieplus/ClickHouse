//
// Created by qieqie on 2021/9/26.
//
#include <Eigen/Core>

#include <DataTypes/DataTypesNumber.h>
#include <Functions/FunctionFactory.h>
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
        if (arguments.size() < 2)
            throw Exception(
                "Function " + getName() + " needs at least two argument; passed " + toString(arguments.size()) + ".",
                ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

        DataTypePtr arguments_type;
        for (size_t index = 0; index < arguments.size(); ++index)
        {
            const DataTypeArray * array_type = checkAndGetDataType<DataTypeArray>(arguments[index].type.get());

            if (!array_type)
                throw Exception(
                    "Argument " + toString(index + 1) + " of function " + getName() + " must be array. Found "
                        + arguments[index].type->getName() + " instead.",
                    ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);

            if (!arguments_type)
            {
                arguments_type = array_type->getNestedType();
            }
            else if (!areTypesEqual(arguments_type, array_type->getNestedType()))
            {
                throw Exception(
                    "Argument " + toString(index + 1) + " of function " + getName() + " must be same type of first argument, which is "
                        + arguments[0].type->getName() + ".",
                    ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
            }
        }
        return std::make_shared<DataTypeArray>(nested_result_type);
    }

    ColumnPtr
    executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr & result_type, size_t /*input_rows_count*/) const override
    {
        size_t num_arguments = arguments.size();
        if (num_arguments < 2)
        {
            throw Exception("At least two Argument required.", ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);
        }

        const auto & return_type = result_type;
        auto res_ptr = return_type->createColumn();
        ColumnArray & res = assert_cast<ColumnArray &>(*res_ptr);
        IColumn & res_data = res.getData();
        ColumnArray::Offsets & res_offsets = res.getOffsets();

        const DataTypeArray * first_column_type = checkAndGetDataType<DataTypeArray>(arguments[0].type.get());
        const DataTypePtr first_column_nested_type = first_column_type->getNestedType();
        const ColumnPtr first_column = arguments[0].column;

        /*
        const ColumnArray * first_column_array = checkAndGetColumn<ColumnArray>(first_column_ptr.get());

        if (!first_column_array) {
            throw Exception(
                "Argument 1 of function " + getName() + " must be array."
                " Found column " + first_column_ptr->getName() + " instead.",
                ErrorCodes::ILLEGAL_COLUMN);
        }
        */

        std::vector<const ColumnPtr> columns;
        for (size_t i = 1; i < num_arguments; ++i)
        {
            columns.emplace_back(arguments[i].column);
        }

        switch (first_column_nested_type->getTypeId())
        {
            case TypeIndex::UInt8:
                executeMatrix<UInt8>(first_column, columns, res_data, res_offsets);
                break;
            case TypeIndex::Float32:
                executeMatrix<Float32>(first_column, columns, res_data, res_offsets);
                break;
            default:
                throw Exception(
                    "Nested array type " + first_column_nested_type->getName() + " is not supported as argument of function arrayDistance.",
                    ErrorCodes::ILLEGAL_COLUMN);
        }

        return res_ptr;
    }

private:
    template <typename T>
    static bool
    executeMatrix(const ColumnPtr first_column, std::vector<const ColumnPtr> columns, IColumn & res_col, ColumnArray::Offsets & res_offsets)
    {
        PaddedPODArray<RT> & res_data = typeid_cast<ColumnVector<RT> &>(res_col).getData();
        Eigen::MatrixX<RT> mx, my;

        if (!columnToMatrix<T>(first_column, &mx))
        {
            throw Exception("First argument of function arrayDistance must be array. ", ErrorCodes::ILLEGAL_COLUMN);
        }
        //std::cout << "matrix x:\n" << mx << std::endl;

        if (!constColumnsToMatrix<T>(columns, &my))
        {
            throw Exception("Arguments of function arrayDistance must be const array. ", ErrorCodes::ILLEGAL_COLUMN);
        }
        //std::cout << "matrix y:\n" << my << std::endl;

        for (auto col : mx.colwise())
        {
            auto norms = (my.colwise() - col).colwise().norm();
            for (auto norm : norms)
            {
                res_data.emplace_back(norm);
            }

            res_offsets.emplace_back(res_data.size());
        }

        return true;
    }

    template <typename T>
    static bool constColumnsToMatrix(std::vector<const ColumnPtr> columns, Eigen::MatrixX<RT> * mat)
    {
        const ColumnPtr & first_column = columns[0];
        const ColumnConst * first_column_const = checkAndGetColumnConst<ColumnArray>(first_column.get());
        if (!first_column_const)
            return false;

        const ColumnArray * first_const_array = checkAndGetColumn<ColumnArray>(first_column_const->getDataColumnPtr().get());
        if (!first_const_array)
            return false;

        const ColumnArray::Offsets & offsets = first_const_array->getOffsets();
        mat->resize(offsets.front(), offsets.size() * columns.size());

        ColumnArray::Offset col = 0;
        for (const ColumnPtr & column : columns)
        {
            const ColumnConst * const_column = checkAndGetColumnConst<ColumnArray>(column.get());
            if (!const_column)
                return false;

            const ColumnArray * const_array = checkAndGetColumn<ColumnArray>(const_column->getDataColumnPtr().get());
            if (!const_array)
                return false;

            const ColumnVector<T> * column_vec = checkAndGetColumn<ColumnVector<T>>(const_array->getData());
            const PaddedPODArray<T> & column_data = column_vec->getData();

            ColumnArray::Offset prev = 0;
            for (auto offset : offsets)
            {
                for (ColumnArray::Offset idx = 0; idx < offset - prev; ++idx)
                {
                    (*mat)(idx, col) = column_data[prev + idx];
                }
                ++col;
                prev = offset;
            }
        }

        return true;
    }

    template <typename T>
    static bool columnToMatrix(const ColumnPtr column, Eigen::MatrixX<RT> * mat)
    {
        const ColumnPtr & column_ptr = column->convertToFullColumnIfConst();
        const ColumnArray * column_array = checkAndGetColumn<ColumnArray>(column_ptr.get());
        if (!column_array)
            return false;

        const ColumnArray::Offsets & offsets = column_array->getOffsets();
        mat->resize(offsets.front(), offsets.size());

        const ColumnVector<T> * column_vec = checkAndGetColumn<ColumnVector<T>>(column_array->getData());
        const PaddedPODArray<T> & column_data = column_vec->getData();

        ColumnArray::Offset col = 0, prev = 0;
        for (auto offset : offsets)
        {
            for (ColumnArray::Offset idx = 0; idx < offset - prev; ++idx)
            {
                (*mat)(idx, col) = column_data[prev + idx];
            }
            ++col;
            prev = offset;
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
