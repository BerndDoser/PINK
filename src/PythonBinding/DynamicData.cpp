/**
 * @file   PythonBinding/DynamicData.cpp
 * @date   Aug 5, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#include "DynamicData.h"

namespace pink {

DynamicData::DynamicData(std::string const& data_type, std::string const& layout, std::vector<ssize_t> shape, void* ptr)
 : data_type(data_type),
   layout(layout),
   dimensionality(shape.size())
{
    if (data_type != "float32") throw std::runtime_error("data-type not supported");
    if (layout != "cartesian-2d") throw std::runtime_error("layout not supported");

    std::vector<uint32_t> my_shape(std::begin(shape), std::end(shape));

    if (dimensionality == 1)
    {
        auto p = static_cast<float*>(ptr);
        std::vector<float> v(p, p + shape[0]);
        data = std::make_shared<Data<CartesianLayout<1>, float>>(
            CartesianLayout<1>{my_shape[0]}, v);
    }
    else if (dimensionality == 2)
    {
        auto p = static_cast<float*>(ptr);
        std::vector<float> v(p, p + shape[0] * shape[1]);

        data = std::make_shared<Data<CartesianLayout<2>, float>>(
            CartesianLayout<2>{my_shape[0], my_shape[1]}, v);
    }
    else if (dimensionality == 3)
    {
        auto p = static_cast<float*>(ptr);
        std::vector<float> v(p, p + shape[0] * shape[1] * shape[2]);
        data = std::make_shared<Data<CartesianLayout<3>, float>>(
            CartesianLayout<3>{my_shape[0], my_shape[1], my_shape[2]}, v);
    }
    else
    {
        throw std::runtime_error("dimensionality not supported");
    }
}

buffer_info DynamicData::get_buffer_info() const
{
    if (dimensionality == 2)
    {
        auto&& shape = std::dynamic_pointer_cast<Data<CartesianLayout<2>, float>>(data)->get_dimension();
        auto&& ptr = std::dynamic_pointer_cast<Data<CartesianLayout<2>, float>>(data)->get_data_pointer();

        return buffer_info(static_cast<void*>(ptr), static_cast<ssize_t>(sizeof(float)), "f", static_cast<ssize_t>(2),
            std::vector<ssize_t>{shape[0], shape[1]},
            std::vector<ssize_t>{sizeof(float) * shape[1], sizeof(float)});
    }
    else
    {
        throw std::runtime_error("dimensionality not supported");
    }
}

} // namespace pink
